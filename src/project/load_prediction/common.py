"""Shared contracts; no GUI or neural runtime imports."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

import numpy as np

Cancel = Callable[[], bool] | None


class Cancelled(RuntimeError):
    pass


def checkpoint(cancel: Cancel = None):
    if cancel and cancel():
        raise Cancelled("任务已取消；已有结果未覆盖")


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    """Atomic replacement, including on cancellation/failure before saving."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(json_value(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        # Windows readers (including the job-status poller or sync client) can
        # hold a brief handle without delete sharing. Keep the old valid JSON
        # until replacement succeeds; never truncate it as a fallback.
        for attempt in range(7):
            try:
                os.replace(temp, path)
                break
            except PermissionError as exc:
                if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == 6:
                    raise
                time.sleep(min(.01 * 2 ** attempt, .1))
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def version_of(value):
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def file_hash(path, cancel: Cancel = None):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint(cancel)
            digest.update(chunk)
    return digest.hexdigest()
