"""Explicit feed execution device and reproducible CUDA identity."""
from __future__ import annotations

import os
import subprocess

CUDA_POLICY = dict(deterministic_algorithms=True, cudnn_deterministic=True,
                   cudnn_benchmark=False, matmul_allow_tf32=False,
                   cudnn_allow_tf32=False, cublas_workspace_config=":4096:8")


def feed_execution_identity(device="cpu"):
    """Describe the enforced policy without changing global process settings."""
    import torch
    if device not in ("cpu", "cuda"):
        raise ValueError("进给训练device只支持显式cpu或cuda")
    result = dict(device=device, torch=str(torch.__version__), torch_cuda=torch.version.cuda,
                  cudnn_version=torch.backends.cudnn.version(),
                  cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads())
    if device == "cpu":
        result["policy"] = dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
        return result
    if not torch.cuda.is_available():
        raise ValueError("显式请求CUDA但当前PyTorch无可用CUDA；拒绝静默回退CPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_POLICY["cublas_workspace_config"]:
        raise ValueError("CUDA进给训练须在启动进程前设置CUBLAS_WORKSPACE_CONFIG=:4096:8")
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total",
                                 "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                text=True, timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("无法记录CUDA硬件与driver身份，拒绝无来源训练") from exc
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(dict(index=index, name=properties.name,
            capability=list(torch.cuda.get_device_capability(index)), total_memory=properties.total_memory,
            multiprocessor_count=properties.multi_processor_count))
    result.update(policy=dict(CUDA_POLICY), selected_index=0, visible_devices=devices,
                  cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  nvidia_smi_inventory=sorted(line.strip() for line in output.splitlines() if line.strip()))
    return result


def configure_feed_device(device="cpu"):
    import torch
    identity = feed_execution_identity(device)
    if device == "cuda":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
    return torch.device(device), identity
