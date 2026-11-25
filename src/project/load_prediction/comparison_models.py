"""Explicit condition-identity encoding for a diagnostic comparison only.

The standard five-branch shape is retained. ``machine.category`` is a storage
slot for condition IDs in this comparison, not a physical machine descriptor.
All other branches stay at their fixed reference. No program or round identity
is used, and unseen condition IDs cannot be inferred or calibrated here.
"""
from __future__ import annotations

from .common import version_of
from .condition import encode_conditions, fit_condition_spec, validate_condition_spec


IDENTITY_MODE = "identity_only_diagnostic"


def _identity_records(records):
    result = []
    for record in records:
        metadata = getattr(record, "metadata", record)
        if not isinstance(metadata, dict):
            raise ValueError("编号工况对照需要含 condition_id 的元数据字典")
        identity = metadata.get("condition_id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("编号工况对照必须显式提供非空字符串 condition_id")
        result.append({"machine": {"category": identity}})
    return result


def fit_identity_condition_spec(records):
    """Fit training IDs only; a single observed level has no trainable branch."""
    spec = fit_condition_spec(_identity_records(records))
    spec.update(encoding_mode=IDENTITY_MODE, identity_field="condition_id")
    spec["version"] = version_of({key: value for key, value in spec.items() if key != "version"})
    return spec


def encode_identity_conditions(records, spec):
    """Encode explicit, known IDs, rejecting missing or unseen identities.

    Removing the diagnostic dispatch marker before using the standard encoder
    keeps its public dispatch from recursing back into this comparison.
    """
    validate_condition_spec(spec)
    if spec.get("encoding_mode") != IDENTITY_MODE or spec.get("identity_field") != "condition_id":
        raise ValueError("工况编号编码需要 identity_only_diagnostic 对照定义")
    standard = {key: value for key, value in spec.items() if key not in ("encoding_mode", "version")}
    standard["version"] = version_of(standard)
    return encode_conditions(_identity_records(records), standard, allow_unknown=False)
