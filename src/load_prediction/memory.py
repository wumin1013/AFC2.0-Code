"""Deterministic, chronological same-position memory (no neural gate)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .common import save_json, version_of


@dataclass
class PositionMemory:
    grid_mm: float = .1
    alpha: float = .3
    minimum_repeats: int = 2
    consistency_floor_w: float = 20.
    entries: dict = field(default_factory=dict)
    events: dict = field(default_factory=dict)
    schema: str = "afc.memory.2"

    def __post_init__(self):
        if self.grid_mm <= 0 or not 0 < self.alpha <= 1 or self.minimum_repeats < 1 or self.consistency_floor_w <= 0:
            raise ValueError("记忆配置无效")

    def _context_key(self, context):
        required = ("program_version", "parameter_group", "tool_session", "main_version", "residual_version")
        if any(not context.get(k) for k in required):
            raise ValueError("同位置记忆缺少程序、工况、刀具会话或模型版本")
        optional = ("condition_id", "tool_id", "process_version", "feed_model_version", "protocol_version")
        return version_of({k: context.get(k) for k in (*required, *optional)})

    def _key(self, context, line, segment_mm, instance):
        return self._context_key(context) + f":{int(line)}:{int(instance)}:{round(segment_mm / self.grid_mm)}"

    @staticmethod
    def _event(context, round_id, event_order):
        event = event_order if event_order is not None else context.get("actual_event_order")
        if event is None:
            if context.get("protocol_version"):
                raise ValueError("正式记忆需要实际采集顺序，不能用计划轮次代替")
            event = round_id  # Explicit legacy diagnostic compatibility only.
        if isinstance(event, bool) or not isinstance(event, (int, np.integer)) or event < 0:
            raise ValueError("实际事件顺序必须为非负整数")
        return int(event)

    def predict(self, context, line, segment_mm, round_id=None, instance=0, *, event_order=None):
        if not context.get("tool_session"):
            return 0.
        event = self._event(context, round_id, event_order)
        if self.events.get(self._context_key(context), -1) >= event:
            raise ValueError("记忆包含当前或未来实际事件，拒绝信息泄漏")
        state = self.entries.get(self._key(context, line, segment_mm, instance))
        if state is None:
            return 0.
        if state["last_event"] >= event:
            raise ValueError("记忆包含当前或未来轮次/实际事件，拒绝信息泄漏")
        if state["count"] < self.minimum_repeats:
            return 0.
        variance = state["variance"]
        gate = self.consistency_floor_w ** 2 / (self.consistency_floor_w ** 2 + variance)
        return float(gate * state["mean"])

    def update_round(self, context, frame, remaining_error_w, round_id=None, *, healthy, no_alarm,
                     complete, aligned, event_order=None):
        if not all(v is True for v in (healthy, no_alarm, complete, aligned)):
            return {"updated": 0, "status": "paused_quality"}
        if not context.get("tool_session"):
            return {"updated": 0, "status": "paused_unknown_session"}
        event = self._event(context, round_id, event_order)
        context_key = self._context_key(context)
        if self.events.get(context_key, -1) >= event:
            raise ValueError("同一实际事件不能重复更新，必须按实际时间顺序更新记忆")
        error = np.asarray(remaining_error_w, float)
        if len(frame) != len(error) or not np.all(np.isfinite(error)):
            raise ValueError("记忆误差长度不匹配或存在无效值")
        grouped = {}
        for (_, row), value in zip(frame.iterrows(), error):
            key = self._key(context, row.line, row.segment_mm, row.execution_instance)
            grouped.setdefault(key, []).append(value)
        # Preflight all keys so an out-of-order update cannot partially mutate.
        if any(key in self.entries and self.entries[key]["last_event"] >= event for key in grouped):
            raise ValueError("同一轮不能重复更新，必须按时间顺序更新记忆")
        for key, values in grouped.items():
            value = float(np.median(values))
            previous = self.entries.get(key)
            if previous is None:
                self.entries[key] = dict(mean=value, variance=0., count=1, last_round=round_id, last_event=event)
            else:
                delta = value - previous["mean"]
                previous.update(mean=previous["mean"] + self.alpha * delta,
                                variance=(1 - self.alpha) * (previous["variance"] + self.alpha * delta ** 2),
                                count=previous["count"] + 1, last_round=round_id, last_event=event)
        if grouped:
            self.events[context_key] = event
        return {"updated": len(grouped), "status": "updated"}

    def reset(self):
        self.entries.clear()
        self.events.clear()

    def save(self, path):
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("schema") != "afc.memory.2":
            raise ValueError("记忆格式不匹配；旧按计划轮次保存的记忆必须冷启动")
        return cls(**payload)
