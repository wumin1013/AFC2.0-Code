"""Load-model devices reuse the established deterministic feed execution policy."""
from __future__ import annotations


def validate_cpu_threads(device, cpu_threads=None, cpu_interop_threads=None):
    for name, value in (("cpu_threads", cpu_threads), ("cpu_interop_threads", cpu_interop_threads)):
        if value is not None and (device != "cpu" or type(value) is not int or value < 1):
            raise ValueError(f"{name}只接受CPU设备的显式正整数")


def request_cpu_threads(request):
    """New load jobs may select threads; old feed/all jobs keep their policy."""
    if request.get("mode") != "load":
        return None, None
    values = set()
    for candidate in request.get("candidates") or []:
        config = candidate.get("load_training", {})
        device = config.get("device", "cpu")
        threads = config.get("cpu_threads"), config.get("cpu_interop_threads")
        validate_cpu_threads(device, *threads)
        if device == "cpu":
            values.add(threads)
    if len(values) > 1:
        raise ValueError("同一load任务的CPU候选必须使用一致的线程配置")
    return next(iter(values), (None, None))


def _configure_cpu_threads(device, cpu_threads=None, cpu_interop_threads=None):
    import torch
    validate_cpu_threads(device, cpu_threads, cpu_interop_threads)
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    if cpu_interop_threads is not None and torch.get_num_interop_threads() != cpu_interop_threads:
        try:
            torch.set_num_interop_threads(cpu_interop_threads)
        except RuntimeError as exc:
            raise ValueError("当前进程已使用其他interop线程配置；须在新工作进程重新准备") from exc


def load_execution_identity(device="cpu", *, cpu_threads=None, cpu_interop_threads=None):
    from .feed_device import feed_execution_identity
    validate_cpu_threads(device, cpu_threads, cpu_interop_threads)
    identity = {"schema": "afc.load_execution.1", **feed_execution_identity(device)}
    for name, value in (("cpu_threads", cpu_threads), ("cpu_interop_threads", cpu_interop_threads)):
        if value is not None:
            if identity[name] != value:
                raise ValueError(f"CPU实际执行线程与请求不一致：{name}")
            identity["requested_" + name] = value
    return identity


def configure_load_device(device="cpu", *, cpu_threads=None, cpu_interop_threads=None):
    from .feed_device import configure_feed_device
    _configure_cpu_threads(device, cpu_threads, cpu_interop_threads)
    resolved, _ = configure_feed_device(device)
    return resolved, load_execution_identity(device, cpu_threads=cpu_threads, cpu_interop_threads=cpu_interop_threads)


def request_load_execution(request):
    """Apply one explicit CPU policy before ready, recovery and worker identity."""
    threads, interop = request_cpu_threads(request)
    _configure_cpu_threads("cpu", threads, interop)
    devices = sorted({c.get("load_training", {}).get("device", "cpu") for c in request.get("candidates") or []})
    return {device: load_execution_identity(device, cpu_threads=threads if device == "cpu" else None,
            cpu_interop_threads=interop if device == "cpu" else None) for device in devices}


def rng_state(device):
    """Only torch RNGs are consumed by the load optimizer and initialization."""
    import torch
    result = {"torch_rng": torch.get_rng_state()}
    if str(device).startswith("cuda"):
        result["cuda_rng"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state, device):
    import torch
    torch.set_rng_state(state["torch_rng"])
    if str(device).startswith("cuda"):
        states = state.get("cuda_rng")
        if not isinstance(states, list) or len(states) != torch.cuda.device_count():
            raise ValueError("CUDA负载恢复缺少匹配的全部设备随机状态")
        torch.cuda.set_rng_state_all(states)
