"""Versioned single-file AFC tensor model with embedded numerical metadata.

This module is copied unchanged into each independently deployed runtime.
Only tensors and JSON-compatible values are stored. Loading uses weights_only.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import os
import tempfile

SCHEMA = "afc.spindle_package.3"
SUPPORTED_SCHEMAS = ("afc.spindle_package.2", SCHEMA)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile_bytes(profile):
    return (json.dumps(profile, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _stored_profile(profile):
    value = deepcopy(profile)
    value.pop("tensor_model_sha256", None)
    for name in ("artifact", "tensor_artifact"):
        if name in value:
            value[name].pop("sha256", None)
    value.get("source_components", {}).get("cutting", {}).pop("tensor_model_sha256", None)
    return value


def _runtime_profile(profile, path, digest, schema=SCHEMA):
    value = deepcopy(profile)
    value["tensor_model_sha256"] = digest
    logical_filename = (value.get("artifact", {}).get("filename")
                        or value.get("tensor_artifact", {}).get("filename")
                        or "SpindlePowerPredictionModel_V1.0.pt")
    for name in ("artifact", "tensor_artifact"):
        value[name] = dict(filename=logical_filename, sha256=digest, kind=schema)
    value.get("source_components", {}).get("cutting", {})["tensor_model_sha256"] = digest
    return value


def _profile_without_storage_names(profile):
    """Older sidecars recorded their host filename rather than a model property."""
    value = deepcopy(profile)
    for name in ("artifact", "tensor_artifact"):
        if name in value:
            value[name].pop("filename", None)
    return value


def state_sha256(state):
    import torch
    digest = hashlib.sha256()
    if not isinstance(state, dict) or not state:
        raise ValueError("Model state must be a nonempty dictionary")
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state must contain named tensors")
        value = tensor.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite model tensor: " + name)
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def numerical_config(profile, base_component=None):
    result = dict(architecture=deepcopy(profile["model_config"]),
                  normalization=deepcopy(profile["normalization"]),
                  base_curve=deepcopy(profile["base_curve"]))
    if base_component is not None:
        result["base_component"] = deepcopy(base_component)
    result["prediction_policy"] = deepcopy(profile.get("prediction_policy", {}))
    return result


def load_package(path, profile_path=None, *, expected_sha256=None, profile_sha256=None):
    import torch
    path = Path(path).resolve(strict=True)
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("Selected model SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    single = isinstance(payload, dict) and payload.get("schema") in SUPPORTED_SCHEMAS
    if isinstance(payload, dict) and str(payload.get("schema", "")).startswith("afc.spindle_package.") and not single:
        raise ValueError("Unsupported spindle package; update the compatible runtime")
    if single:
        state = payload["state_dict"]
        profile = _runtime_profile(payload["profile"], path, actual, payload["schema"])
        base = payload["base_component"]
        if state_sha256(state) != payload["state_sha256"]:
            raise ValueError("Embedded tensor state digest differs")
        metadata = dict(profile=payload["profile"], base_component=base)
        if hashlib.sha256(profile_bytes(metadata)).hexdigest() != payload["metadata_sha256"]:
            raise ValueError("Embedded model metadata digest differs")
        profile_digest = hashlib.sha256(profile_bytes(profile)).hexdigest()
        if profile_path:
            external = json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
            if _profile_without_storage_names(external) != _profile_without_storage_names(profile):
                raise ValueError("External profile differs from embedded model configuration")
            # Preserve explicit legacy request identities while checking all
            # numerical metadata against the selected single-file package.
            profile = external
            profile_digest = file_sha256(profile_path)
    else:
        if not profile_path:
            raise ValueError("Legacy tensor state requires its original matching profile")
        state = payload
        profile = json.loads(Path(profile_path).read_text(encoding="utf-8-sig"))
        profile_digest = file_sha256(profile_path)
        base = None
        if profile.get("tensor_model_sha256") != actual:
            raise ValueError("Legacy profile binds a different model")
        state_sha256(state)
    if profile.get("schema") != "afc.spindle_tensor_profile.1":
        raise ValueError("Unsupported numerical model profile")
    if profile_sha256 and profile_digest != profile_sha256:
        raise ValueError("Model profile SHA differs")
    return dict(state=state, profile=profile, base_component=base,
                model_sha256=actual, profile_sha256=profile_digest, is_single_file=single)


def save_package(path, state, profile, base_component):
    import torch
    path = Path(path)
    if path.exists():
        raise FileExistsError("Model destination already exists: " + str(path))
    if not isinstance(base_component, dict):
        raise ValueError("The complete base component is required for single-file export")
    stored = _stored_profile(profile)
    metadata = dict(profile=stored, base_component=deepcopy(base_component))
    tensors = {name: value.detach().cpu().clone() for name, value in state.items()}
    payload = dict(schema=SCHEMA, state_dict=tensors, **metadata,
                   state_sha256=state_sha256(tensors),
                   metadata_sha256=hashlib.sha256(profile_bytes(metadata)).hexdigest())
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".writing", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        restored = load_package(temporary)
        if state_sha256(restored["state"]) != payload["state_sha256"]:
            raise ValueError("Single-file model readback differs")
        # Windows rename refuses replacement, preserving an independently published model.
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path.resolve()), sha256=file_sha256(path), bytes=path.stat().st_size)


def extract_profile(model_path, destination):
    """Write an internal cache for existing parsers; users select only the model."""
    package = load_package(model_path)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    value = profile_bytes(package["profile"])
    if target.exists():
        if target.read_bytes() != value:
            raise ValueError("Existing profile cache belongs to another model")
    else:
        with target.open("xb") as stream:
            stream.write(value)
    return target
