"""Persistent single-worker local jobs. Browser lifetime never owns training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import uuid

from .common import Cancelled, save_json
from .training_service import TrainingService

TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_text(path):
    """Retry only short Windows sharing/access conflicts; never hide bad data."""
    for attempt in range(7):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 6:
                raise
            time.sleep(min(.01 * 2 ** attempt, .1))


def _worker(directory):
    directory = Path(directory)
    request = json.loads(_read_text(directory / "request.json"))
    service = TrainingService(request.pop("service_output"))
    state = json.loads(_read_text(directory / "state.json"))
    state.update(status="running", started_at=_now(), pid=os.getpid())
    save_json(directory / "state.json", state)

    def progress(event):
        event = {"time": _now(), **event}
        with (directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        state["progress"] = event
        save_json(directory / "state.json", state)

    try:
        if request.get("mode") == "load":
            state["load_execution"] = service._run_identity(request)["environment"]["load_execution"]
            save_json(directory / "state.json", state)
        result = service.execute(request, cancel=lambda: (directory / "cancel").exists(), progress=progress)
        state.update(status="succeeded", result=result)
    except Cancelled as exc:
        state.update(status="cancelled", error=str(exc))
    except BaseException as exc:
        state.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    state["finished_at"] = _now()
    save_json(directory / "state.json", state)


class JobService:
    def __init__(self, training_service=None, *, recover=True):
        self.service = training_service or TrainingService()
        self.directory = self.service.output_dir / "jobs"
        self.directory.mkdir(parents=True, exist_ok=True)
        self._owner = (self.directory / "service.lock").open("a+b")
        self._owner.seek(0)
        if not self._owner.read(1):
            self._owner.write(b"0")
            self._owner.flush()
        self._owner.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._owner.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._owner.close()
            raise RuntimeError("该输出目录已有运行中的任务服务，请通过现有网页/API创建任务") from exc
        self.lock = threading.RLock()
        self.active = None
        self.closed = threading.Event()
        if recover:
            for path in self.directory.glob("*/state.json"):
                state = json.loads(_read_text(path))
                if state["status"] not in TERMINAL:
                    state.update(status="interrupted", error="服务停止后未完成；可使用 resume_from_job_id 显式恢复已保存epoch", finished_at=_now())
                    save_json(path, state)
        self.thread = threading.Thread(target=self._schedule, daemon=True, name="afc-job-scheduler")
        self.thread.start()

    def create(self, request):
        if self.closed.is_set():
            raise ValueError("任务服务已关闭")
        self.service.validate_recovery(request)
        preview = (self.service.experiment({**request, "execute": False}) if request.get("mode") == "experiment" else
                   self.service.preview(request["snapshot_id"], request.get("candidates"), request.get("mode", "all")))
        job_id = uuid.uuid4().hex
        directory = self.directory / job_id
        directory.mkdir()
        request = {**request, "job_id": job_id, "service_output": str(self.service.output_dir)}
        save_json(directory / "request.json", request)
        state = dict(job_id=job_id, status="queued", created_at=_now(), preview=preview, progress=None,
                     resume_from_job_id=request.get("resume_from_job_id"))
        save_json(directory / "state.json", state)
        return state

    def get(self, job_id):
        if not job_id.isalnum():
            raise ValueError("任务ID不合法")
        path = self.directory / job_id / "state.json"
        if not path.is_file():
            raise ValueError("任务不存在")
        state = json.loads(_read_text(path))
        log = path.parent / "events.jsonl"
        state["events"] = [json.loads(x) for x in _read_text(log).splitlines()[-500:]] if log.exists() else []
        return state

    def list(self):
        return [json.loads(_read_text(p)) for p in sorted(self.directory.glob("*/state.json"))]

    def cancel(self, job_id):
        with self.lock:
            state = self.get(job_id)
            if state["status"] in TERMINAL:
                return state
            directory = self.directory / job_id
            (directory / "cancel").touch()
            if state["status"] == "queued":
                state.pop("events", None)
                state.update(status="cancelled", finished_at=_now())
                save_json(directory / "state.json", state)
            return self.get(job_id)

    def _schedule(self):
        while not self.closed.wait(.25):
            with self.lock:
                if self.active:
                    job_id, process, output = self.active
                    if process.poll() is None:
                        continue
                    output.close()
                    state = self.get(job_id)
                    if state["status"] not in TERMINAL:
                        state.pop("events", None)
                        state.update(status="interrupted", error=f"工作进程异常退出: {process.returncode}", finished_at=_now())
                        save_json(self.directory / job_id / "state.json", state)
                    self.active = None
                queued = [s for s in self.list() if s["status"] == "queued"]
                if not queued:
                    continue
                state = min(queued, key=lambda s: s["created_at"])
                directory = self.directory / state["job_id"]
                output = (directory / "worker.log").open("ab")
                env = dict(os.environ)
                env.update(PYTHONUTF8="1", MKL_THREADING_LAYER="SEQUENTIAL", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get("PYTHONPATH", "")
                try:
                    request = json.loads(_read_text(directory / "request.json"))
                    from .load_device import request_cpu_threads
                    cpu_threads, _ = request_cpu_threads(request)
                    if cpu_threads is not None:
                        env["OMP_NUM_THREADS"] = str(cpu_threads)
                    process = subprocess.Popen([sys.executable, "-m", "load_prediction.job_service", "--worker", str(directory)],
                        env=env, stdout=output, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    state.update(status="running", started_at=_now(), pid=process.pid)
                    save_json(directory / "state.json", state)
                    self.active = (state["job_id"], process, output)
                except (OSError, ValueError) as exc:
                    output.close()
                    state.update(status="failed", error=str(exc), finished_at=_now())
                    save_json(directory / "state.json", state)

    def shutdown(self):
        if self.closed.is_set():
            return
        self.closed.set()
        self.thread.join(timeout=3)
        with self.lock:
            if self.active:
                job_id, process, output = self.active
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
                output.close()
                self.active = None
            for state in self.list():
                if state["status"] not in TERMINAL:
                    state.update(status="interrupted", error="服务关闭，任务未完成", finished_at=_now())
                    save_json(self.directory / state["job_id"] / "state.json", state)
            self._owner.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self._owner.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._owner.fileno(), fcntl.LOCK_UN)
            self._owner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    _worker(parser.parse_args().worker)
