"""Public neural-model API; torch is only imported when explicitly requested."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ModelConfig:
    coefficient_widths: tuple = (32, 32)
    condition_dim: int = 0
    vc_reference_m_s: float = 1.
    fz_reference_mm: float = .1
    feed_reference_mm_min: float = 2000.
    hidden: int = 32
    dilations: tuple = (1, 2, 4)
    kernel_size: int = 3
    attention_heads: int = 4
    attention_layers: int = 2
    window_points: int = 256
    condition_spec: dict | None = None
    condition_width: int = 16
    quantiles: tuple = (.05, .5, .95)
    physical_mode: str = "local"
    use_condition_effects: bool = True
    use_residual_modulation: bool = True
    residual_output: str = "quantiles"
    condition_input: str = "descriptors"
    coefficient_condition: str = "explicit"
    route: str = "predicted_feed_v1"
    schema: str = "afc.load.3"

    def validate(self):
        if self.route not in ("predicted_feed_v1", "nominal_programmed_residual_v1"):
            raise ValueError("未知负载训练路线")
        if self.physical_mode not in ("local", "fixed") or self.residual_output not in ("quantiles", "point"):
            raise ValueError("物理与残差对照配置不受支持")
        if self.condition_input not in ("descriptors", "identity_only_diagnostic") or self.coefficient_condition not in ("explicit", "implicit"):
            raise ValueError("工况对照输入或系数结构不受支持")
        if self.coefficient_condition == "implicit" and self.physical_mode != "local":
            raise ValueError("隐式工况对照采用局部系数网络")
        if type(self.use_condition_effects) is not bool or type(self.use_residual_modulation) is not bool:
            raise ValueError("工况调制开关必须为布尔值")
        if self.schema != "afc.load.3":
            raise ValueError("旧负载模型输入/工况/分位格式不兼容，需按实际进给重新训练")
        if self.condition_dim != 0:
            raise ValueError("不接受未声明工况向量；使用五分支condition_spec物理描述")
        if type(self.condition_width) is not int or self.condition_width < 1:
            raise ValueError("工况分支宽度必须为正整数")
        if tuple(self.quantiles) != (.05, .5, .95):
            raise ValueError("当前分位定义固定为0.05、0.5、0.95")
        if self.condition_spec is not None:
            from .condition import validate_condition_spec
            validate_condition_spec(self.condition_spec)
            if self.condition_spec.get("encoding_mode", "descriptors") != self.condition_input:
                raise ValueError("工况编码与配置来源不一致")
        if self.attention_heads < 1 or self.hidden < 4 or self.hidden % self.attention_heads or self.hidden % 2:
            raise ValueError("hidden 必须为偶数且能被 attention_heads 整除")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0 or not self.dilations or min(self.dilations) < 1:
            raise ValueError("TCN kernel_size 为不小于3的奇数，dilations 为正数")
        if min(self.vc_reference_m_s, self.fz_reference_mm, self.feed_reference_mm_min) <= 0 or self.condition_dim < 0 or self.window_points < 2 or self.attention_layers < 1:
            raise ValueError("模型配置无效")
        if not self.coefficient_widths or min(self.coefficient_widths) < 1:
            raise ValueError("系数网络至少包含一个正宽度隐层")


def _runtime():
    # A host embedding AFC may have initialized NumPy before project.shared.
    # Fail recoverably instead of accepting multiple OpenMP runtimes.
    import sys
    if sys.platform == "win32":
        from threadpoolctl import threadpool_info
        active = threadpool_info()
        if any(i.get("internal_api") == "mkl" and i.get("threading_layer") == "intel" for i in active):
            raise RuntimeError("当前进程已加载Intel线程版MKL。请设置 MKL_THREADING_LAYER=SEQUENTIAL 后重启AFC，再加载神经模型。")
    try:
        from . import neural
        return neural
    except (ImportError, OSError, AttributeError) as exc:
        raise RuntimeError("神经模型依赖不可用。请在 AFC 环境按 requirements-load-prediction.txt 安装 PyTorch；其他模块仍可使用。") from exc


def build_load_models(config=None, *, device="cpu"):
    config = config or ModelConfig()
    config.validate()
    return _runtime().ModelBundle(config).to(device)


def load_models(path, *, device="cpu"):
    return _runtime().ModelBundle.load(path).to(device)


def predict_cutting_power(bundle, frame, *, condition=None, memory=None, context=None, round_id=None,
                          feed_mode=None, calibration=None, interval_calibrator=None):
    return _runtime().predict_cutting_power(bundle, frame, condition=condition, memory=memory,
                                            context=context, round_id=round_id, feed_mode=feed_mode,
                                            calibration=calibration, interval_calibrator=interval_calibrator)
