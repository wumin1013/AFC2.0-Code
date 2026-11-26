"""Independent postoperative reference display; no label or score mutations."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np
import pandas as pd

KEYS = ('line', 'execution_instance', 'segment_mm', 'path_mm')

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def bound(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError(f'Artifact hash changed: {path}')
    return path

def load_overlay(path, package):
    spec = importlib.util.spec_from_file_location(f'{package}.reference_alignment', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def build_reference_attachment(parent, measurement, base, metadata, oracle, align):
    """Align frozen raw reference on the parent NC grid; test against old core."""
    if metadata.get('position_offset_mm') is None:
        raise ValueError('Reference attachment requires the frozen parent coordinate offset')
    speed = np.asarray(measurement.values['spindle_speed_rpm'], float)
    power = np.asarray(measurement.values['spindle_power_w'], float)
    baseline = np.asarray(base['power_w'], float)
    correction = np.asarray(base.get('reference_correction_w', np.zeros(len(power))), float)
    usable = np.asarray(base['valid'], bool) & np.isfinite(power) & np.isfinite(baseline) & np.isfinite(speed) & np.isfinite(correction)
    line_offset = metadata.get('measurement_line_offset', 0)
    mapping = {int(v): int(v) + line_offset for v in np.unique(measurement.values['program_line_raw']) if np.isfinite(v)}
    channels = {'reference_power_cut_w': power - baseline,
                'reference_base_power_w': baseline, 'reference_correction_w': correction}
    kwargs = dict(valid=usable, line_map=mapping, position_offset_mm=metadata['position_offset_mm'])
    original, original_info = oracle(parent, measurement, channels, **kwargs)
    result, info = align(parent, measurement, channels, support_provenance=True, **kwargs)
    pd.testing.assert_frame_equal(result[original.columns], original, check_exact=True)
    assert {k: v for k, v in info.items() if k != 'support_segments'} == original_info
    available = result.aligned.to_numpy(bool) & np.isfinite(result.reference_power_cut_w.to_numpy(float))
    attachment = parent[list(KEYS)].copy(deep=True)
    attachment.insert(0, 'grid_index', np.arange(len(parent), dtype=np.int64))
    attachment['reference_power_cut_w'] = result.reference_power_cut_w.where(available)
    attachment['reference_aligned'] = available
    attachment['reference_support_segment_id'] = result.support_segment_id.where(available, -1).to_numpy(np.int64)
    attachment['reference_alignment_error_mm'] = result.alignment_error_mm.where(available)
    label = base.get('hybrid_label_metadata', base.get('label_metadata', {}))
    attachment['reference_kind'] = label.get('reference_kind', 'postoperative_measured_idle')
    attachment['reference_evidence_grade'] = label.get('evidence_grade', 'observed_platform_with_background_assumption')
    finite_parent = np.isfinite(parent.power_cut_w.to_numpy(float))
    if not available[finite_parent].all():
        raise ValueError('Independent reference lost a finite parent reference')
    np.testing.assert_array_equal(parent.loc[finite_parent, 'power_cut_w'], attachment.loc[finite_parent, 'reference_power_cut_w'])
    return attachment, info, {'historical_alignment_exact': True,
        'finite_parent_reference_points': int(finite_parent.sum()), 'finite_parent_reference_exact': True,
        'raw_base_valid': int(np.asarray(base['valid'], bool).sum()), 'raw_usable': int(usable.sum())}

def continuous_reference_runs(frame):
    """Never bridge missing grid rows, support identities, visits or reversals."""
    needed = {'grid_index', *KEYS, 'reference_power_cut_w', 'reference_aligned', 'reference_support_segment_id'}
    if needed - set(frame):
        raise ValueError(f'Reference attachment missing columns: {sorted(needed - set(frame))}')
    available = frame.reference_aligned.to_numpy(bool) & np.isfinite(frame.reference_power_cut_w.to_numpy(float))
    support = frame.reference_support_segment_id.to_numpy(np.int64)
    available &= support >= 0
    grid = frame.grid_index.to_numpy(np.int64)
    x = frame.path_mm.to_numpy(float)
    same = ((support[1:] == support[:-1]) & (np.diff(grid) == 1) & (np.diff(x) > 0)
            & (frame.line.to_numpy()[1:] == frame.line.to_numpy()[:-1])
            & (frame.execution_instance.to_numpy()[1:] == frame.execution_instance.to_numpy()[:-1]))
    starts = np.flatnonzero(available & ~np.r_[False, available[:-1] & same])
    ends = np.flatnonzero(available & ~np.r_[available[1:] & same, False])
    return [np.arange(a, b + 1) for a, b in zip(starts, ends)]
