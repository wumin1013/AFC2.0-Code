"""Feed-only reuse contract. Full-run/checkpoint recovery remains stricter."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from .common import file_hash
from .feed_samples import FEED_FEATURE_COLUMNS
from .training_checkpoint import digest

SCHEMA = "afc.feed_reuse.1"
# These modules implement feed features, labels, windows, normalization and learning.
# Load labels, base qualification, condition/residual networks and service dispatch
# are deliberately absent; their full source hashes remain in the run provenance.
DEPENDENCIES = ("feed_identity.py", "feed_device.py", "feed_training.py", "feed_model.py", "feed_neural.py",
                "feed_samples.py", "feed.py", "measurement.py", "sample_rules.py",
                "experiment_protocol.py", "common.py")
METADATA = ("record_id", "round_id", "feature", "program_id", "dataset_role",
            "program_version", "machine_version", "machine_settings_version", "feed_override",
            "machining_mode", "coordinate_mode", "spacing_mm", "schema", "feature_columns",
            "sample_rules_version", "process_mapping_rules_version", "measurement_content_sha256",
            "measurement_sha256", "measurement_source", "position_source", "position_columns",
            "measured_feed_source", "position_offset_mm", "complete", "round_complete",
            "alignment_reliable", "no_alarm", "tool_healthy", "cutting_contact_confirmed")
EXTRA_COLUMNS = ("line", "execution_instance", "segment_mm", "path_mm", "sequence_id",
                 "motion", "cutting_contact", "ap_mm", "ae_mm", "pause", "known_break",
                 "feed_actual_mm_min", "valid_label")


def feed_source_fingerprints():
    root = Path(__file__).parent
    return {**{name: file_hash(root / name) for name in DEPENDENCIES},
            "shared/process_path.py": file_hash(root.parent / "process_path.py")}


def feed_sample_identity(samples):
    digest_value = hashlib.sha256()
    members = []
    for sample in samples:
        columns = list(FEED_FEATURE_COLUMNS) + [c for c in EXTRA_COLUMNS if c in sample.frame and c not in FEED_FEATURE_COLUMNS]
        frame = sample.frame.loc[:, columns]
        metadata = {key: sample.metadata[key] for key in METADATA if key in sample.metadata}
        digest_value.update(digest(dict(metadata=metadata, columns=columns,
                                       dtypes=[str(value) for value in frame.dtypes])).encode("ascii"))
        digest_value.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
        members.append({key: sample.metadata.get(key) for key in ("record_id", "round_id", "feature", "dataset_role")})
    return dict(sha256=digest_value.hexdigest(), members=members)


def feed_training_identity(train, validation, *, training_config, model_config, full=False):
    import torch
    from .feed_device import feed_execution_identity
    return dict(schema=SCHEMA, train=feed_sample_identity(train), validation=feed_sample_identity(validation or []),
                configuration=dict(training=asdict(training_config), model=asdict(model_config), full=bool(full)),
                code=feed_source_fingerprints(), environment=dict(python=platform.python_version(),
                    torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__,
                    execution=feed_execution_identity(training_config.device),
                    torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                    threading={key: os.environ.get(key) for key in
                               ("MKL_THREADING_LAYER", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}))
