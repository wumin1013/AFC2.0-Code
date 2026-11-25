"""Content-bound epoch checkpoints; no automatic migration or changed-data resume."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile

import pandas as pd

from .common import file_hash, json_value, save_json

SCHEMA = "afc.training_checkpoint.1"


def digest(value):
    return hashlib.sha256(json.dumps(json_value(value), ensure_ascii=False, sort_keys=True,
                                    allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def source_fingerprints(directory=None):
    directory = Path(__file__).parent if directory is None else Path(directory)
    source_root = directory.parent
    historical = source_root.name == "project"
    analysis = source_root if historical else source_root / "interval_analysis"
    common = source_root if historical else source_root / "afc_common"

    fingerprints = {p.name: file_hash(p) for p in sorted(directory.glob("*.py"))}
    fingerprints["shared/process_path.py"] = file_hash(common / "process_path.py")
    fingerprints["shared/processing_core.py"] = file_hash(analysis / "processing_core.py")
    fingerprints.update({
        "shared/segmentation/" + p.name: file_hash(p)
        for p in sorted((analysis / "segmentation").glob("*.py"))
    })
    return fingerprints


def sample_fingerprint(samples):
    """Hash actual prepared values, column order/types and provenance, not filenames."""
    result = hashlib.sha256()
    for sample in samples:
        frame = sample.frame
        result.update(digest(dict(metadata=sample.metadata, columns=list(frame.columns),
                                  dtypes=[str(t) for t in frame.dtypes])).encode("ascii"))
        result.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return result.hexdigest()


def training_identity(kind, train, validation, configuration, *, context=None):
    import numpy as np
    import torch
    execution = {}
    if kind == "feed":
        from .feed_device import feed_execution_identity
        execution["feed_execution"] = feed_execution_identity(configuration["training"].get("device", "cpu"))
    return dict(kind=kind, train=sample_fingerprint(train), validation=sample_fingerprint(validation),
                configuration=json_value(configuration), context=json_value(context),
                code=source_fingerprints(), environment=dict(python=platform.python_version(),
                torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__, **execution))


def save_checkpoint(path, identity, state):
    if path is None:
        return
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(dict(schema=SCHEMA, fingerprint=digest(identity), identity=identity,
                            state=state), stream)
            stream.flush()
            os.fsync(stream.fileno())
        epoch = state.get("epoch")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("检查点必须包含非负整数epoch")
        archive = path.parent / "epochs" / path.stem
        archive.mkdir(parents=True, exist_ok=True)
        saved_epoch = archive / f"epoch_{epoch:04d}.pt"
        index_path = archive / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else dict(
            schema="afc.epoch_archive.1", fingerprint=digest(identity), stage=path.stem, epochs=[])
        if index["fingerprint"] != digest(identity):
            raise ValueError("epoch归档身份不符，拒绝覆盖历史")
        if saved_epoch.exists():
            # A frozen main-only artifact may be exported again during an
            # explicit run recovery; identical bytes do not create a new epoch.
            if (state.get("artifact_schema") == "afc.main_baseline.1" and index["epochs"]
                    and index["epochs"][-1]["epoch"] == epoch
                    and file_hash(saved_epoch) == file_hash(temporary) == index["epochs"][-1]["sha256"]):
                return
            raise FileExistsError("epoch检查点已经存在，拒绝改写历史")
        if any(item["epoch"] >= epoch for item in index["epochs"]):
            raise ValueError("epoch归档顺序重复，拒绝覆盖历史")
        # Cloud-synced Windows folders can reject hard links (WinError 396).
        # Publish the archive by rename, then keep a separate atomic latest copy.
        os.rename(temporary, saved_epoch)
        audit = dict(schema=SCHEMA, fingerprint=digest(identity), sha256=file_hash(saved_epoch),
                     epoch=epoch, completed=state.get("completed", False),
                     stage=path.stem, archive=str(saved_epoch.relative_to(path.parent)))
        save_json(saved_epoch.with_suffix(".pt.json"), audit)
        index["epochs"].append(audit)
        save_json(index_path, index)
        shutil.copyfile(saved_epoch, temporary)
        os.replace(temporary, path)
        save_json(path.with_suffix(path.suffix + ".json"), audit)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, identity, *, resume=False):
    if path is None:
        return None
    path = Path(path)
    companion = path.with_suffix(path.suffix + ".json")
    if not path.exists():
        if resume and (companion.exists() or (path.parent / "epochs" / path.stem).exists()):
            raise ValueError("阶段latest检查点缺失但回执或epoch归档仍存在，拒绝静默重训")
        return None
    if not resume:
        raise FileExistsError("检查点已存在；必须显式请求恢复，不能覆盖历史训练")
    import torch
    if not companion.is_file():
        raise ValueError("检查点缺少完整性记录")
    audit = json.loads(companion.read_text(encoding="utf-8"))
    expected = digest(identity)
    if audit.get("schema") != SCHEMA or audit.get("fingerprint") != expected:
        raise ValueError("恢复指纹不一致：数据、配置、源码或环境已变化")
    if audit.get("sha256") != file_hash(path):
        raise ValueError("检查点文件哈希不一致，拒绝恢复")
    if audit.get("archive"):
        archive_path = path.parent / audit["archive"]
        index = json.loads((archive_path.parent / "index.json").read_text(encoding="utf-8"))
        if index.get("fingerprint") != expected or not index.get("epochs") or index["epochs"][-1] != audit:
            raise ValueError("epoch归档索引与阶段检查点不一致，拒绝恢复")
        for entry in index["epochs"]:
            archived = path.parent / entry["archive"]
            if (file_hash(archived) != entry["sha256"] or
                    json.loads(archived.with_suffix(".pt.json").read_text(encoding="utf-8")) != entry):
                raise ValueError("epoch归档文件或回执被修改，拒绝恢复")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA or payload.get("fingerprint") != expected or digest(payload.get("identity")) != expected:
        raise ValueError("检查点内部指纹不一致")
    return payload["state"]
