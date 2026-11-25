"""Resolve explicitly relocated artifacts without rewriting historical evidence.

The relocation receipt establishes locations only. Callers must still verify
each artifact against its original content hash before using its evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from .common import file_hash


def _lexical_absolute(value, base_dir=None):
    """Interpret historical paths with their source OS, never the current OS."""
    text = str(value)
    if not text or "\x00" in text:
        raise ValueError("工件路径为空或含无效字符")
    windows = PureWindowsPath(text)
    path = windows if windows.drive or "\\" in text else PurePosixPath(text)
    if ".." in path.parts:
        raise ValueError("工件路径不得通过上级目录越界")
    if path.is_absolute():
        return path
    if path.drive or path.root:
        raise ValueError("工件路径不能使用盘符相对或无盘符根路径")
    if base_dir is None:
        raise ValueError("相对工件路径缺少来源目录")
    base = _lexical_absolute(base_dir)
    return base.joinpath(*path.parts)


def _local_absolute(value, base_dir=None):
    path = _lexical_absolute(value, base_dir)
    native = isinstance(path, PureWindowsPath) if os.name == "nt" else isinstance(path, PurePosixPath)
    if not native:
        raise ValueError("外部系统工件路径未登记迁移映射")
    return Path(str(path)).resolve()


class ArtifactResolver:
    def __init__(self, reference=None, *, base_dir=None):
        self.reference = deepcopy(reference)
        self.base_dir = _lexical_absolute(base_dir) if base_dir is not None else None
        self.moves = []
        if reference is None:
            return
        if not isinstance(reference, dict) or not reference.get("path") or not reference.get("sha256"):
            raise ValueError("迁移路径映射必须绑定既有清单路径和SHA-256")
        receipt = self._absolute(reference["path"], self.base_dir)
        if not receipt.is_file() or file_hash(receipt) != reference["sha256"]:
            raise ValueError("迁移清单不存在或SHA-256不匹配")
        rows = json.loads(receipt.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("迁移清单必须是明确的来源/目标列表")
        sources, destinations = set(), set()
        for row in rows:
            if not isinstance(row, dict) or row.get("status") != "moved_verified":
                raise ValueError("只接受已完成核验的迁移条目")
            source = _lexical_absolute(row["source"])
            destination = _local_absolute(row["destination"])
            target_key = _lexical_absolute(destination)
            if source in sources or target_key.is_relative_to(source):
                raise ValueError("迁移来源重复或目标位于原目录内")
            if destination in destinations:
                raise ValueError("多个迁移来源指向同一目标，存在冲突")
            for previous_source, previous_destination in self.moves:
                if destination.is_relative_to(previous_destination):
                    implied = previous_source.joinpath(*destination.relative_to(previous_destination).parts)
                    if implied != source:
                        raise ValueError("迁移目标重叠但来源层级不一致，存在冲突")
                elif previous_destination.is_relative_to(destination):
                    implied = source.joinpath(*previous_destination.relative_to(destination).parts)
                    if implied != previous_source:
                        raise ValueError("迁移目标重叠但来源层级不一致，存在冲突")
            sources.add(source)
            destinations.add(destination)
            self.moves.append((source, destination))
        self.moves.sort(key=lambda pair: len(pair[0].parts), reverse=True)
        self.reference = {**reference, "path": str(receipt)}

    @staticmethod
    def _absolute(value, base_dir):
        return _local_absolute(value, base_dir)

    def resolve(self, value, *, base_dir=None):
        base = base_dir if base_dir is not None else self.base_dir
        path = _lexical_absolute(value, base)
        for source, destination in self.moves:
            if path.is_relative_to(source):
                result = destination.joinpath(*path.relative_to(source).parts).resolve()
                if not result.is_relative_to(destination):
                    raise ValueError("迁移解析结果越出已核验目标目录")
                return result
        return _local_absolute(path)


def archived_load_directory(source):
    """Locate the model source in a current or historical source archive."""
    source_dir = Path(source) / "src"
    current = source_dir / "load_prediction"
    return current if current.is_dir() else source_dir / "project" / "load_prediction"
