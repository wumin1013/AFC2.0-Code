"""Build, export or recover the trainable source of a complete AFC sequence model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from spindle_power.spindle_runtime.package import file_sha256, profile_bytes
    from spindle_power.spindle_runtime import load_model
except ModuleNotFoundError as error:
    if error.name != "spindle_power.spindle_runtime.package":
        raise
    # The pre-reorganization tree keeps the identical numerical runtime here.
    from spindle_runtime.package import file_sha256, profile_bytes
    from spindle_runtime import load_model

CONTRACT = "afc.spindle_sequence_onnx.1"
PROFILE_FILE = "afc.profile.json"
SOURCE_FILE = "afc.training.pt"
SCHEMA_FILE = "afc.sequence_contract.json"
INPUT_NAMES = [
    "features", "feature_valid", "source_geometry", "segment_mm", "query_path_mm",
    "command_rpm", "ap_ae_mm", "event_anchor", "event_line", "event_rpm", "extrapolate",
]
OUTPUT_NAMES = [
    "source_power", "source_valid", "trace_power", "trace_valid", "trace_positions",
    "trace_event_elapsed_s", "event_status",
]
EVENT_INPUTS = ["event_anchor", "event_line", "event_rpm"]
TRACE_OUTPUTS = ["trace_power", "trace_valid", "trace_positions", "trace_event_elapsed_s"]


def new_path(value):
    path = Path(value)
    if path.exists():
        raise FileExistsError("Choose a new destination: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build(source, destination):
    from spindle_power.spindle_runtime.sequence_tensor import SpindleSequenceTensor
    from spindle_power.spindle_runtime.sequence_tensor_cutting import SequenceCutting
    from spindle_power.spindle_runtime.sequence_tensor_trace import SequenceTrace

    from spindle_power.spindle_runtime.dynamic_onnx import DynamicBaseONNX
    from spindle_power.spindle_runtime.neural_onnx import CuttingExport

    model, profile = load_model(source)
    execution = profile["execution"]
    policy = profile["prediction_policy"]
    if (policy.get("runtime_contract") != "afc.spindle_runtime.1.2"
            or policy.get("position_policy") != "direct_process_source_positions_v2"
            or execution["window_points"] != 256
            or execution["residual_window_policy"] != "prospective_v1"
            or execution["residual_batch_strategy"] != "padded" or execution["residual_batch_windows"] != 128 or model.dynamic_base is None):
        raise ValueError("This exporter requires the V1.0 direct-position model and its prospective 256-point windows")
    cutting = SequenceCutting(CuttingExport(model), model.base_speeds_rpm, model.base_powers_w)
    trace = SequenceTrace(DynamicBaseONNX(model.dynamic_base), model.dynamic_base.startup_domain_rpm)
    sequence = SpindleSequenceTensor(cutting, trace, model.base_speeds_rpm, model.base_powers_w).eval()
    artifact = torch.jit.script(sequence)
    wrapped = {"schema": CONTRACT, "source_training_pt_sha256": file_sha256(source)}
    torch.jit.save(
        artifact, str(new_path(destination)),
        _extra_files={PROFILE_FILE: profile_bytes(profile), SOURCE_FILE: source.read_bytes(),
                      SCHEMA_FILE: json.dumps(wrapped, sort_keys=True)},
    )
    print("Inference PT", file_sha256(destination), flush=True)


def read_torchscript(path, *, include_source=False):
    entries = {PROFILE_FILE: b"", SCHEMA_FILE: b""}
    if include_source:
        entries[SOURCE_FILE] = b""
    model = torch.jit.load(str(path), map_location="cpu", _extra_files=entries).eval()
    info = json.loads(entries[SCHEMA_FILE])
    profile = json.loads(entries[PROFILE_FILE])
    if (info["schema"] != CONTRACT or info["source_training_pt_sha256"] != profile["tensor_model_sha256"]
            or profile_bytes(profile) != entries[PROFILE_FILE]):
        raise ValueError("Invalid embedded sequence provenance")
    if include_source and hashlib.sha256(entries[SOURCE_FILE]).hexdigest() != info["source_training_pt_sha256"]:
        raise ValueError("Embedded source PT digest differs")
    return model, entries, info


def example():
    length = 16
    features = torch.zeros(length, 20, dtype=torch.float64)
    features[:, :6] = torch.tensor([1., 3., 1500., 10000., 10., 3.], dtype=torch.float64)
    features[:, 10] = 1
    geom = torch.zeros(length, 5, dtype=torch.int64)
    geom[:, 0] = torch.arange(length)
    geom[:, 1] = 7
    geom[:, 4] = 1
    segment = torch.arange(length, dtype=torch.float64)
    return (
        features, torch.ones(length, dtype=torch.bool), geom, segment, segment,
        features[:, 3].contiguous(), features[:, :2].contiguous(), torch.tensor([0, -1]),
        torch.tensor([6, 8]), torch.tensor([10000., 5000.], dtype=torch.float64), torch.tensor([True]),
    )


def export(inference_pt, destination):
    import onnx
    import onnxruntime as ort

    model, entries, info = read_torchscript(inference_pt)
    axis = {name: {0: "N"} for name in INPUT_NAMES[:7] + OUTPUT_NAMES[:2]}
    axis.update({name: {0: "E"} for name in EVENT_INPUTS + ["event_status"]})
    axis.update({name: {0: "T"} for name in TRACE_OUTPUTS})
    sample = example()
    output = new_path(destination)
    with torch.inference_mode():
        torch.onnx.export(
            model, sample, str(output), input_names=INPUT_NAMES, output_names=OUTPUT_NAMES, dynamic_axes=axis,
            dynamo=False, opset_version=17, external_data=False, do_constant_folding=True,
        )
    graph = onnx.load(str(output))
    profile = entries[PROFILE_FILE]
    onnx.helper.set_model_props(graph, {
        "afc.contract": CONTRACT,
        "afc.profile": profile.decode("utf-8"),
        "afc.profile_sha256": hashlib.sha256(profile).hexdigest(),
        "afc.source_training_pt_sha256": info["source_training_pt_sha256"],
        "afc.source_inference_pt_sha256": file_sha256(inference_pt),
    })
    onnx.checker.check_model(graph, full_check=True)
    onnx.save_model(graph, str(output), save_as_external_data=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = model(*sample)
    received = session.run(None, dict(zip(INPUT_NAMES, (value.numpy() for value in sample), strict=True)))
    for name, got, expected in zip(OUTPUT_NAMES, received, reference, strict=True):
        if got.dtype.kind in "biu":
            np.testing.assert_array_equal(got, expected.numpy(), err_msg=name)
        else:
            np.testing.assert_allclose(got, expected.numpy(), atol=.002, rtol=2e-6, equal_nan=True, err_msg=name)
    print("ONNX", file_sha256(output), flush=True)


def recover(inference_pt, destination):
    _, entries, _ = read_torchscript(inference_pt, include_source=True)
    new_path(destination).write_bytes(entries[SOURCE_FILE])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    make = subcommands.add_parser("build")
    make.add_argument("--training-pt", type=Path, required=True)
    make.add_argument("--output", type=Path, required=True)
    for operation in ("export", "recover-training"):
        command = subcommands.add_parser(operation)
        command.add_argument("--pt", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "build":
        build(args.training_pt, args.output)
    elif args.command == "export":
        export(args.pt, args.output)
    else:
        recover(args.pt, args.output)


if __name__ == "__main__":
    main()
