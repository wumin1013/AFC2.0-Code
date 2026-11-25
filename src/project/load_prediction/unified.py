"""Portable initial model and read-only inference for the local model host.

The interval EXE does not import this module. A shared model contains immutable
initial components; a task supplies its own memory and completed-data updates.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .common import file_hash, json_value, save_json, version_of

SCHEMA = "afc.unified_spindle.1"
CONTEXT_DEFAULT = {
    "schema_version": "1.0",
    "machine": {"model": "200C-2"},
    "tool": {"name": "硬质铝合金三刃立铣刀D10", "diameter_mm": 10.0, "flutes": 3},
    "workpiece": {"material": "6061铝"},
}
POSITION_KEYS = ["line", "execution_instance", "segment_mm"]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _base_from_bytes(data, runtime_dir):
    """Reuse the original loader and its validation without a persistent sidecar."""
    from .base_power import BasePowerModel
    directory = Path(runtime_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="afc_base_", dir=directory) as temporary:
        path = Path(temporary) / "base.json"
        path.write_bytes(data)
        return BasePowerModel.load(path)


def _cutting_from_bytes(data, device):
    from .models import _runtime
    return _runtime().ModelBundle.load(io.BytesIO(data)).to(device)


class UnifiedSpindleModel:
    """Single .pt containing exact source components and portable metadata."""

    def __init__(self, payload, *, runtime_dir, device="cpu"):
        import torch
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("统一模型格式不兼容")
        manifest = payload.get("manifest", {})
        if payload.get("package_version") != version_of(manifest):
            raise ValueError("统一模型清单版本不匹配")
        for name in ("cutting", "base"):
            data = payload.get(name)
            if not isinstance(data, bytes) or _sha(data) != manifest["components"][name]["sha256"]:
                raise ValueError(f"{name}组件内容与SHA-256不匹配")
        for name, data in payload.get("evidence", {}).items():
            if _sha(data) != manifest.get("evidence_sha256", {}).get(name):
                raise ValueError(f"证据{name}内容不匹配")
        if set(payload.get("evidence", {})) != set(manifest.get("evidence_sha256", {})):
            raise ValueError("证据清单不完整")
        self.payload = deepcopy(payload)
        self._protected_paths = set()
        # Model construction initializes tensors before restoring the frozen
        # weights; do not advance the embedding host's CPU RNG as a side effect.
        with torch.random.fork_rng(devices=[]):
            self.cutting = _cutting_from_bytes(payload["cutting"], device)
        self.base = _base_from_bytes(payload["base"], runtime_dir)
        cutting = manifest["components"]["cutting"]
        if (self.cutting.current_version() != cutting["version"]
                or self.cutting.main_version != cutting["main_version"]
                or self.cutting.residual_version != cutting["residual_version"]):
            raise ValueError("切削组件版本不匹配")
        base = manifest["components"]["base"]
        if self.base.version != base["version"] or self.base.status != base["status"]:
            raise ValueError("基础功率组件版本或资格不匹配")

    @property
    def version(self):
        return self.payload["package_version"]

    @property
    def manifest(self):
        return deepcopy(self.payload["manifest"])

    @classmethod
    def load(cls, path, *, runtime_dir=None, device="cpu"):
        import torch
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(payload, runtime_dir=runtime_dir or path.parent, device=device)
        model._protected_paths.add(path.resolve())
        return model

    @classmethod
    def build(cls, cutting_path, base_path, *, runtime_dir, evidence_paths=()):
        from .memory import PositionMemory
        cutting_data, base_data = Path(cutting_path).read_bytes(), Path(base_path).read_bytes()
        cutting = _cutting_from_bytes(cutting_data, "cpu")
        base = _base_from_bytes(base_data, runtime_dir)
        if cutting.config.route != "nominal_programmed_residual_v1":
            raise ValueError("当前统一入口仅实现已选名义物理残差路线")
        if any(branch.get("active") for branch in cutting.config.condition_spec["branches"].values()):
            raise ValueError("当前默认工况适配器仅用于已冻结的无活动材质分支模型")
        evidence = {}
        for item in evidence_paths:
            path = Path(item)
            if path.name in evidence:
                raise ValueError("打包证据不能重名")
            evidence[path.name] = path.read_bytes()
        identity = cutting.metadata.get("posthoc_extension", {})
        manifest = {
            "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
            "components": {
                "cutting": {"sha256": _sha(cutting_data), "version": cutting.current_version(),
                    "main_version": cutting.main_version, "residual_version": cutting.residual_version,
                    "schema": "afc.load.3", "identity": {k: identity.get(k) for k in ("hidden", "fold", "seed", "epoch")}},
                "base": {"sha256": _sha(base_data), "version": base.version, "schema": base.schema,
                    "status": base.status},
            },
            "evidence_sha256": {name: _sha(value) for name, value in evidence.items()},
            "input_context_default": deepcopy(CONTEXT_DEFAULT),
            "input_definition": cutting.input_definition(),
            "normalization": deepcopy(cutting.normalization),
            "condition_encoding": deepcopy(cutting.config.condition_spec),
            "memory_initial_snapshot": asdict(PositionMemory()),
            "memory_policy": "empty_initial_snapshot_then_task_isolated_external_state",
            "calibration_initial_snapshot": {"condition": None, "interval": None, "status": "not_calibrated"},
            "applicability": {"machine": "200C-2", "diameter_mm": 10.0, "flutes": 3,
                "workpiece_record": "6061铝", "tool_record": CONTEXT_DEFAULT["tool"]["name"],
                "material_fields_used_by_network": False, "cross_combination_validated": False},
            "prediction_policy": {"default_base_mode": "steady_curve_research",
                "steady_assumption": "commanded speed settled; not a verified dynamic timeline",
                "interval": "cutting_q05_q95_plus_base_point_estimate",
                "includes_base_uncertainty": False, "total_interval_calibrated": False},
            "research_only": True, "production_ready": False, "independent_test": False,
            "runtime_required": "AFC load_prediction runtime; PyTorch 2.x; NumPy 2.x",
        }
        model = cls(dict(schema=SCHEMA, manifest=manifest, package_version=version_of(manifest),
                        cutting=cutting_data, base=base_data, evidence=evidence), runtime_dir=runtime_dir)
        model._protected_paths = {Path(item).resolve() for item in (cutting_path, base_path, *evidence_paths)}
        return model

    def save(self, path):
        import torch
        path = Path(path)
        if path.resolve() in self._protected_paths:
            raise ValueError("模型输出不得覆盖输入模型、基础组件或来源证据，请使用新版本文件名")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
        os.close(fd)
        try:
            torch.save(self.payload, name)
            # Validate before replacing an existing published package.
            checked = self.load(name, runtime_dir=path.parent)
            if checked.version != self.version:
                raise ValueError("统一模型保存读回不一致")
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)
        return {"path": str(path.resolve()), "sha256": file_hash(path), "version": self.version}

    def with_base(self, base_path, *, runtime_dir):
        """Return a new package; never mutate shared source or cutting bytes."""
        payload = deepcopy(self.payload)
        data = Path(base_path).read_bytes()
        model = _base_from_bytes(data, runtime_dir)
        payload["base"] = data
        manifest = payload["manifest"]
        manifest["parent_version"] = self.version
        manifest["created_at"] = datetime.now(timezone.utc).isoformat()
        manifest["components"]["base"] = dict(sha256=_sha(data), version=model.version,
            status=model.status, schema=model.schema)
        manifest["base_update_evidence_status"] = "new_component_qualification_must_be_reviewed"
        payload["package_version"] = version_of(manifest)
        updated = type(self)(payload, runtime_dir=runtime_dir)
        updated._protected_paths = self._protected_paths | {Path(base_path).resolve()}
        return updated


def read_context(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema_version") != "1.0":
        raise ValueError("machining_context.json需要schema_version=1.0")
    if any(not isinstance(value.get(key), dict) for key in ("machine", "tool", "workpiece")):
        raise ValueError("工况文件需要machine、tool、workpiece三个对象")
    machine, tool, workpiece = value["machine"], value["tool"], value["workpiece"]
    if machine.get("model") != "200C-2":
        raise ValueError("当前初始模型只登记200C-2；其他机床待适配")
    if (isinstance(tool.get("diameter_mm"), bool) or tool.get("diameter_mm") != 10
            or isinstance(tool.get("flutes"), bool) or tool.get("flutes") != 3):
        raise ValueError("当前初始模型仅支持已记录D10、3刃组合")
    if tool.get("name") != CONTEXT_DEFAULT["tool"]["name"] or workpiece.get("material") != "6061铝":
        raise ValueError("刀具或工件记录与当前初始组合不一致，需先适配验证")
    return value


def prepare_inputs(nc_path, process_path, context, bundle):
    """Reuse frozen NC/grid/geometry features, without a measured-data input."""
    from .feed import FeedPlan, MachineConfig
    from .feed_samples import prepare_feed_features
    from .measurement import detect_encoding
    from .process_coordinates import PROCESS_COORDINATE_RULES_VERSION, prepare_data_supported_geometry
    from .samples import load_input_view, read_process_table
    from .experiment_protocol import PROTOCOL_VERSION

    task = context.get("task", {})
    known = dict(program_id=task.get("program_id") or Path(nc_path).stem,
        program_version=None, machine_id="200C-2", machine_settings_version="recorded_execution_settings_semantics_uninterpreted",
        feed_override=1.0, machining_mode="recorded_execution_settings_semantics_uninterpreted",
        coordinate_mode="workpiece_cutting", diameter_mm=10.0, teeth=3,
        parameter_group=task.get("parameter_group", "P0"), tool_session=task.get("tool_session"),
        round_id=task.get("round_id"), actual_event_order=task.get("actual_event_order"),
        protocol_version=PROTOCOL_VERSION, position_policy="command_only_v1",
        process_mapping_rules_version=PROCESS_COORDINATE_RULES_VERSION,
        condition_descriptors=deepcopy(bundle.config.condition_spec["reference_descriptors"]))
    features = prepare_feed_features(nc_path, MachineConfig(name="200C-2"),
        program_id=known["program_id"], feed_override=known["feed_override"],
        machining_mode=known["machining_mode"], machine_settings_version=known["machine_settings_version"],
        coordinate_mode="workpiece_cutting")
    plan = FeedPlan(features.frame, features.metadata["program_version"], features.metadata["machine_version"], False, {})
    known["program_version"] = plan.program_version
    process = read_process_table(process_path, plan)
    distances = process.source_path_mm.to_numpy(float)
    if not np.isfinite(distances).all() or (distances < 0).any() or (np.diff(distances) < 0).any():
        raise ValueError("当前原工艺s列需有限、非负、非递减累计行程；禁止按增量二次累加")
    tokens = {}
    with Path(process_path).open(encoding=detect_encoding(process_path)) as stream:
        header = next(stream)
        if not (header.startswith("序号") and "MRR" in header and "s(mm)" in header):
            raise ValueError("当前498模型需原始序号/N/ap/ae/F/MRR/s/S/G工艺导出；其他格式需对应位置规则验证")
        for number, line in enumerate(stream, 2):
            if line.strip():
                tokens[number] = line.split(maxsplit=8)[6]
    geometry, source = prepare_data_supported_geometry(process, plan, known, source_distance_tokens=tokens)
    return load_input_view(geometry, "programmed"), source


def _source_predictions(source, grid):
    """Map only inside adjacent supported points of one NC execution segment."""
    fields = ["base_power_w", "cutting_power_w", "spindle_power_w", "spindle_lower_w", "spindle_upper_w"]
    rows = []
    groups = {key: group.sort_values("segment_mm") for key, group in grid.groupby(["line", "execution_instance"], sort=False)}
    for source_index, (_, point) in enumerate(source.iterrows()):
        row = {"source_process_row_index": source_index, "source_index": source_index,
            "line": int(point.line), "execution_instance": int(point.execution_instance),
            "segment_mm": float(point.segment_mm), "source_file_line": int(point.source_file_line),
            "source_cell_start_mm": float(point.segment_start_mm),
            "source_cell_end_mm": float(point.source_cell_end_mm),
            "source_path_mm": float(point.source_path_mm),
            "nc_path_mm": float(point.nc_path_mm),
            "mapping_method": "unavailable", "prediction_segment_mm": None,
            "prediction_status": "unavailable_input"}
        for field in fields:
            row[field] = None
        group = groups.get((point.line, point.execution_instance))
        if group is not None and np.isfinite(point.segment_mm):
            positions = group.segment_mm.to_numpy(float)
            right = int(np.searchsorted(positions, point.segment_mm, side="left"))
            if right < len(group):
                left = right if abs(positions[right] - point.segment_mm) <= 1e-10 else right - 1
                if left >= 0:
                    pair = group.iloc[[left, right]]
                    pair_contact = (pair.ap_mm > 0) & (pair.ae_mm > 0)
                    if (pair.prediction_status.eq("research_prediction").all()
                            and pair.sequence_id.nunique() == 1
                            and pair_contact.eq(point.ap_mm > 0 and point.ae_mm > 0).all()
                            and np.isfinite(pair[fields].to_numpy(float)).all()):
                        for field in fields:
                            row[field] = float(np.interp(point.segment_mm, positions[[left, right]], pair[field].to_numpy(float)))
                        row["prediction_status"] = "research_prediction"
                        row["mapping_method"] = "exact_nc_grid_point" if left == right else "within_sequence_linear_interpolation"
                        row["prediction_segment_mm"] = float(point.segment_mm)
            if row["prediction_status"] == "unavailable_input" and np.isfinite([point.segment_start_mm, point.source_cell_end_mm]).all():
                # A source point owns its preceding geometric cell. At a
                # support/sequence boundary use only an existing prediction
                # inside this exact cell with the same contact state. This is
                # a cell representative, not inference at a fabricated node.
                inside = (group.segment_mm > point.segment_start_mm) & (group.segment_mm <= point.source_cell_end_mm)
                contact = (group.ap_mm > 0) & (group.ae_mm > 0)
                same_contact = contact == (point.ap_mm > 0 and point.ae_mm > 0)
                candidates = group.loc[inside & same_contact & group.prediction_status.eq("research_prediction")]
                candidates = candidates.loc[np.isfinite(candidates[fields].to_numpy(float)).all(axis=1)]
                if not candidates.empty:
                    candidate = candidates.iloc[int(np.argmin(abs(candidates.segment_mm.to_numpy(float) - point.segment_mm)))]
                    between = group.loc[(group.segment_mm >= min(candidate.segment_mm, point.segment_mm))
                        & (group.segment_mm <= max(candidate.segment_mm, point.segment_mm))]
                    if (between.prediction_status.eq("research_prediction").all()
                            and between.sequence_id.eq(candidate.sequence_id).all()):
                        for field in fields:
                            row[field] = float(candidate[field])
                        row.update(prediction_status="research_prediction", mapping_method="source_cell_nearest_supported_grid",
                            prediction_segment_mm=float(candidate.segment_mm))
        row["total_power_w"] = row.pop("spindle_power_w")
        row["total_lower_w"] = row.pop("spindle_lower_w")
        row["total_upper_w"] = row.pop("spindle_upper_w")
        row["status"] = row["prediction_status"]
        rows.append(row)
    return rows


def predict_files(model, *, nc_path, process_path, context_path, output_dir, memory=None,
                  calibration=None, interval_calibrator=None, dynamic_request=None):
    """Three-file AFO API. Returns research spindle estimates and their scope.

    Without a verified uniform time mapping, only a settled-command baseline
    estimate is emitted. It is never labelled a qualified dynamic prediction.
    """
    from .models import predict_cutting_power
    from .neural import prospective_residual_context
    from .short_gap_bridge import bridge_short_prediction_gaps
    from .total_power import _dynamic_base
    directory = Path(output_dir)
    input_paths = {Path(path).resolve() for path in (nc_path, process_path, context_path)} | model._protected_paths
    if input_paths & {(directory / name).resolve() for name in ("prediction.csv", "prediction.json")}:
        raise ValueError("预测输出不得覆盖NC、原始工艺、工况或模型输入文件")
    context = read_context(context_path)
    references = {name: {"name": Path(path).name, "sha256": file_hash(path)} for name, path in
                  (("nc", nc_path), ("process", process_path), ("context", context_path))}
    sample, source = prepare_inputs(nc_path, process_path, context, model.cutting)
    selected, indices, _ = prospective_residual_context(sample.frame, model.cutting.config.window_points)
    if selected.empty:
        raise ValueError("NC/工艺没有完整可预测位置")
    cutting_mask = (selected.ap_mm > 0) & (selected.ae_mm > 0) & (selected.motion != 0)
    if selected.loc[cutting_mask, "spindle_rpm"].nunique() > 1:
        raise ValueError("当前切削模型未验证切削中主轴变速")
    result = predict_cutting_power(model.cutting, selected, context=sample.metadata,
        memory=memory, round_id=sample.metadata.get("round_id"), calibration=calibration,
        interval_calibrator=interval_calibrator)
    speeds = selected.spindle_rpm.to_numpy(float)
    if not np.isfinite(speeds).all() or (speeds < model.base.speeds_rpm[0]).any() or (speeds > model.base.speeds_rpm[-1]).any():
        raise ValueError("主轴转速超出基础功率曲线支持范围，禁止外推")
    if dynamic_request is None:
        base = np.interp(speeds, model.base.speeds_rpm, model.base.powers_w)
        base_mode = "steady_curve_research_estimate"
        timing = {"assumption": "commanded_speed_settled_not_verified", "dynamic_timeline_available": False}
    else:
        domain = dict(spindle_rpm_min=model.base.speeds_rpm[0], spindle_rpm_max=model.base.speeds_rpm[-1])
        base, timing = _dynamic_base(selected, dynamic_request, domain, model.base)
        base_mode = "commanded_dynamic_research_estimate"
    output = sample.frame.copy()
    output["prediction_status"] = "unavailable_input"
    output.loc[indices, "prediction_status"] = "research_prediction"
    for name, value in result.items():
        if isinstance(value, np.ndarray) and value.shape == (len(selected),):
            output[name] = np.nan
            output.loc[indices, name] = value
    for name, value in (("base_power_w", base), ("spindle_power_w", base + result["cutting_power_w"]),
                        ("spindle_lower_w", base + result["lower_w"]), ("spindle_upper_w", base + result["upper_w"])):
        output[name] = np.nan
        output.loc[indices, name] = value
    for name, path in (("nc", nc_path), ("process", process_path), ("context", context_path)):
        if file_hash(path) != references[name]["sha256"]:
            raise ValueError("输入文件在预测期间改变")
    directory.mkdir(parents=True, exist_ok=True)
    # Per-task output is caller-owned. No training or memory write occurs here.
    destination = directory / "prediction.csv"
    fd, temporary = tempfile.mkstemp(prefix="prediction_", suffix=".tmp", dir=directory)
    os.close(fd)
    try:
        output.to_csv(temporary, index=False, encoding="utf-8", float_format="%.17g")
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    receipt = dict(schema="afc.spindle_prediction.1", model_version=model.version,
        cutting_version=model.cutting.current_version(), base_version=model.base.version,
        base_source_status=model.base.status, base_mode=base_mode, timing=timing,
        input_files=references, machining_context=context, model_inputs=sample.metadata,
        rows=len(output), predicted_rows=len(selected), missing_rows=len(output)-len(selected),
        research_only=True, production_ready=False, independent_test=False,
        interval_kind="cutting_prediction_q05_q95_plus_base_point_estimate",
        interval_scope="cutting_component_only_base_estimate_added", total_interval_calibrated=False,
        quantiles=[.05, .5, .95], includes_base_uncertainty=False,
        memory_mode="cold_start" if memory is None else "caller_task_snapshot_read_only",
        default_material_information_role="record_and_applicability_check_not_new_trained_input",
        prediction_csv={"name": destination.name, "sha256": file_hash(destination)},
        created_at=datetime.now(timezone.utc).isoformat())
    rows = _source_predictions(source, output)
    rows, bridge_metadata = bridge_short_prediction_gaps(rows, output)
    receipt["short_gap_bridge"] = bridge_metadata
    receipt.update(source_rows=len(rows), source_predicted_rows=sum(row["prediction_status"] == "research_prediction" for row in rows),
        position_mapping="source_process_cells_mapped_from_nc_grid", source_index_base=0,
        process_distance_semantics="cumulative_mm", process_distance_validation="finite_nonnegative_nondecreasing",
        source_cell_representative_rows=sum(row["mapping_method"] == "source_cell_nearest_supported_grid" for row in rows),
        process_file_sha256=references["process"]["sha256"], nc_file_sha256=references["nc"]["sha256"],
        program_name=Path(nc_path).stem, interval_status="research_cutting_interval_base_point_added",
        total_power_status="research_estimate_not_qualified_total_prediction",
        source_index_definition="original_process_data_row_order_excluding_header_and_empty_lines",
        line_definition="one_based_physical_nc_line", optimization_qualification="research_only_requires_AFO_integration_validation")
    save_json(directory / "prediction.json", dict(schema="afc.spindle_prediction.1", metadata=receipt, rows=rows))
    return receipt
