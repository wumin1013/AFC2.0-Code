"""Shared tensor model and software-side full-position prediction."""
from .package import load_package, numerical_config
from .model import SpindlePowerPredictionModel


def load_model(path, *, device='cpu'):
    """Load the selected single-file artifact without changing any parameters."""
    import torch
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    bundle = load_package(path)
    model = SpindlePowerPredictionModel(numerical_config(bundle['profile'], bundle['base_component']))
    model.load_state_dict(bundle['state'], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, bundle['profile']


def predict_files(context_path, nc_path, process_path, model_path, *, device='cpu'):
    """Predict source-position power from three business inputs using PT or ONNX.

    Returns a DataFrame and metadata; invalid cutting geometry remains blocked.
    V1.2 adds startup events to metadata, separate from original process rows.
    Cloud model entry points use tensors only, not this file-parsing function.
    """
    import json
    from pathlib import Path
    context = json.loads(Path(context_path).read_text(encoding='utf-8-sig'))
    if Path(model_path).suffix.lower() == '.onnx':
        from .onnx_runtime import predict_onnx_positions
        return predict_onnx_positions(nc_path, process_path, context, model_path, device=device)
    from .full_position import predict_full_positions
    model, profile = load_model(model_path, device=device)
    return predict_full_positions(nc_path, process_path, context, model, profile)


__all__ = ['load_model', 'load_package', 'numerical_config', 'SpindlePowerPredictionModel', 'predict_files']
