"""Paper comparisons as explicit requests to the shared training service.

No model optimization lives here. A dispatch callback must implement the
requested ablation; unsupported modes never become successful experiments.
"""
from __future__ import annotations

from copy import deepcopy

from .common import checkpoint
from .experiment_protocol import PROTOCOL_VERSION

EXPERIMENT_SCHEMA = "afc.paper_experiments.1"


def _spec(name, title, variant, *, multi=False, target=False, adaptation=False, frozen=False):
    required = ["development_data_ready"]
    if multi:
        required.append("multicondition_data_ready")
    if target:
        required.append("independent_target_test_ready")
    if adaptation:
        required.append("target_adaptation_ready")
    if frozen:
        required.append("frozen_source_model_ready")
    return dict(id=name, title=title, variant=variant, requires=required,
                target_budget="same_adaptation_budget" if adaptation else "zero" if target else None)


_EXPERIMENTS = (
    _spec("fixed_physical", "固定剪切/刃口系数物理基线", {"main": "fixed_coefficients", "residual": False, "memory": False}),
    _spec("local_physical", "局部变系数物理基线", {"main": "local_coefficients", "condition": "none", "residual": False, "memory": False}),
    _spec("no_condition", "不使用工况输入", {"condition": "none"}),
    _spec("descriptor_condition", "物理描述工况输入", {"condition": "physical_descriptors"}, multi=True),
    _spec("id_condition", "仅工况编号的诊断对照", {"condition": "identity_only_diagnostic"}, multi=True),
    _spec("implicit_condition", "隐式工况影响", {"condition": "physical_descriptors", "multipliers": "implicit"}, multi=True),
    _spec("explicit_multipliers", "显式剪切/刃口倍率", {"condition": "physical_descriptors", "multipliers": "explicit"}, multi=True),
    _spec("no_residual_modulation", "关闭残差工况幅值与偏置", {"residual_modulation": False}, multi=True),
    _spec("residual_modulation", "开启残差工况幅值与偏置", {"residual_modulation": True}, multi=True),
    _spec("no_memory", "冻结初始同位置记忆", {"memory": False}),
    _spec("memory", "按冻结规则更新同位置记忆", {"memory": True, "memory_order": "predict_score_then_update"}),
    _spec("point_output", "点预测损失对照", {"output": "point", "interval_status": "unavailable"}),
    _spec("quantile_output", "三分位预测", {"output": "quantiles", "quantiles": [.05, .5, .95]}),
    _spec("programmed_feed", "同一冻结负载模型使用编程进给", {"feed_source": "programmed_feed_diagnostic", "freeze_load_model": True, "memory": False}, frozen=True),
    _spec("measured_feed", "同一冻结负载模型使用实测实际进给", {"feed_source": "measured_actual_feed", "freeze_load_model": True, "memory": False}, frozen=True),
    _spec("predicted_feed", "同一冻结负载模型使用BiLSTM进给", {"feed_source": "bilstm_actual_feed", "freeze_load_model": True, "memory": False}, frozen=True),
    _spec("zero_shot", "新工况零目标数据", {"adaptation": "none", "target_fit_records": 0}, multi=True, target=True, frozen=True),
    _spec("four_parameter_calibration", "冻结共享网络的四参数校准", {"adaptation": "four_parameters", "trainable": ["delta_s", "delta_e", "delta_r", "delta_b"]}, multi=True, target=True, adaptation=True, frozen=True),
    _spec("physical_refit", "等预算物理系数重标定", {"adaptation": "physical_coefficients"}, multi=True, target=True, adaptation=True, frozen=True),
    _spec("output_finetune", "等预算输出层微调", {"adaptation": "output_heads_only"}, multi=True, target=True, adaptation=True, frozen=True),
    _spec("full_retrain", "目标工况完整重训练参考", {"adaptation": "full_network_retrain", "reference_not_equal_budget": True}, multi=True, target=True, adaptation=True),
)


def list_experiments(context=None):
    """Explicitly distinguish absent real data from unimplemented dispatchers."""
    context = context or {}
    result = []
    for spec in _EXPERIMENTS:
        item = deepcopy(spec)
        missing = [key for key in spec["requires"] if context.get(key) is not True]
        if "multicondition_data_ready" in missing:
            status = "awaiting_multicondition_data"
        elif missing:
            status = "awaiting_data"
        elif context.get("supported_experiments") is not None and spec["id"] not in context["supported_experiments"]:
            status = "awaiting_service_implementation"
        else:
            status = "dispatch_ready"
        item.update(status=status, missing=missing, schema=EXPERIMENT_SCHEMA,
                    protocol_version=PROTOCOL_VERSION, reports_real_effect=False)
        result.append(item)
    return result


def run_experiment(name, runner, context, *, execute=False, cancel=None, progress=None):
    """Dispatch ``runner(request, cancel=..., progress=...)``; never retrain here.

    The runner is a shared TrainingService adapter. It must fail explicitly if
    it cannot apply a variant, rather than run the default model under its name.
    """
    entries = {item["id"]: item for item in list_experiments(context)}
    if name not in entries:
        raise ValueError(f"未注册的论文对照：{name}")
    spec = entries[name]
    request = {"experiment": name, "ablation": deepcopy(spec["variant"]),
               "protocol_version": PROTOCOL_VERSION, "snapshot_id": context.get("snapshot_id"),
               "source_model_id": context.get("source_model_id"), "seed": context.get("seed", 42),
               "target_adaptation_snapshot": context.get("target_adaptation_snapshot"),
               "independent_test_snapshot": context.get("independent_test_snapshot"),
               "target_budget": deepcopy(context.get("target_budget")),
               "model_config": deepcopy(context.get("model_config", {})),
               "training_config": deepcopy(context.get("training_config", {})),
               "freeze_test_roles": True}
    report = {"schema": EXPERIMENT_SCHEMA, "experiment": name, "status": spec["status"],
              "missing": spec["missing"], "request": request, "real_effect_established": False}
    if spec["status"] != "dispatch_ready":
        return report
    if not execute:
        report["status"] = "prepared_not_executed"
        return report
    if not callable(runner):
        raise ValueError("执行论文对照需要共享TrainingService runner")
    if not request["snapshot_id"]:
        raise ValueError("论文对照必须绑定不可变数据快照")
    if "target_adaptation_ready" in spec["requires"] and not request["target_budget"]:
        raise ValueError("适配对照必须固定目标数据预算，禁止自动使用最终测试记录")
    checkpoint(cancel)
    if progress:
        progress({"phase": "experiment_dispatch", "experiment": name, "ablation": request["ablation"]})
    try:
        result = runner(deepcopy(request), cancel=cancel, progress=progress)
    except NotImplementedError as exc:
        report.update(status="awaiting_service_implementation", reason=str(exc))
        return report
    checkpoint(cancel)
    if not isinstance(result, dict) or result.get("applied_experiment") != name:
        raise ValueError("共享服务必须返回 applied_experiment，证明实际执行了指定对照")
    if result.get("status") not in ("succeeded", "completed", "frozen", "evaluated"):
        report.update(status=result.get("status", "failed"), result=result)
        return report
    report.update(status="completed", result=result)
    # Numerical completion is not by itself a scientific superiority finding.
    return report
