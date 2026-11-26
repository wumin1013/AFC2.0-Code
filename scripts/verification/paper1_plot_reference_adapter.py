"""Paper-1 reference display adapter with explicit bound evidence and honest gaps."""
from pathlib import Path
import numpy as np
import pandas as pd
from paper1_reference_attachment import KEYS, bound, continuous_reference_runs, read_json, sha

REFERENCE_COLUMNS = ('reference_power_cut_w', 'reference_aligned', 'reference_support_segment_id',
                     'reference_alignment_error_mm', 'reference_kind', 'reference_evidence_grade')

def attach_reference(prediction_frame, receipt_path, *, snapshot_path, sample_path, prediction_reference=None):
    """Keep the legacy frozen-prediction API; development attachments need an explicit prediction hash.

    snapshot_path/sample_path name the reference's parent basis. A later frozen
    prediction can be attached only if its complete target and scoring domain
    are exactly that basis; its model identity remains the caller's responsibility.
    """
    receipt_hash = sha(receipt_path)
    receipt = read_json(receipt_path)
    if receipt['status'] != 'candidate_verified_not_formally_released':
        raise ValueError('Unexpected candidate reference receipt status')
    if Path(snapshot_path).resolve() != Path(receipt['snapshot']['path']).resolve():
        raise ValueError('Reference snapshot differs from the plotted prediction')
    if Path(sample_path).resolve() != Path(receipt['sample']['path']).resolve():
        raise ValueError('Reference sample differs from the plotted prediction')
    recorded_prediction = receipt.get('prediction')
    if recorded_prediction and prediction_reference and prediction_reference != recorded_prediction:
        raise ValueError('Cannot replace a prediction bound by the reference receipt')
    prediction_ref = prediction_reference or recorded_prediction
    if not prediction_ref:
        raise ValueError('Development reference requires an explicit frozen prediction reference')
    for key in ('snapshot', 'sample', 'attachment', 'support_segments'):
        bound(receipt[key]['path'], receipt[key]['sha256'])
    bound(prediction_ref['path'], prediction_ref['sha256'])
    sealed_prediction = pd.read_csv(prediction_ref['path'], float_precision='round_trip')
    if set(sealed_prediction) - set(prediction_frame):
        raise ValueError('Plotted frame omits frozen prediction columns')
    pd.testing.assert_frame_equal(prediction_frame[list(sealed_prediction)], sealed_prediction, check_exact=True)
    parent = pd.read_csv(receipt['sample']['path'], float_precision='round_trip')
    if len(parent) != len(prediction_frame):
        raise ValueError('Frozen prediction lost parent grid rows')
    np.testing.assert_array_equal(parent.power_cut_w, prediction_frame.power_cut_w)
    np.testing.assert_array_equal(parent.valid_residual, prediction_frame.scored)
    reference = pd.read_csv(receipt['attachment']['path'], float_precision='round_trip')
    if len(reference) != len(prediction_frame):
        raise ValueError('Reference grid length differs')
    np.testing.assert_array_equal(reference.grid_index, np.arange(len(prediction_frame)))
    for key in KEYS:
        np.testing.assert_array_equal(reference[key], prediction_frame[key])
        np.testing.assert_array_equal(reference[key], parent[key])
    finite_parent = np.isfinite(parent.power_cut_w.to_numpy(float))
    np.testing.assert_array_equal(reference.loc[finite_parent, 'reference_power_cut_w'], parent.loc[finite_parent, 'power_cut_w'])
    if not reference.loc[finite_parent, 'reference_aligned'].all():
        raise ValueError('Reference attachment hides a finite parent reference')
    output = prediction_frame.copy(deep=True)
    output['grid_index'] = reference.grid_index
    for key in REFERENCE_COLUMNS:
        output[key] = reference[key].to_numpy()
    # Existing target, scoring mask and model outputs must survive attachment.
    original_columns = [c for c in prediction_frame if c not in (*REFERENCE_COLUMNS, 'grid_index')]
    pd.testing.assert_frame_equal(output[original_columns], prediction_frame[original_columns], check_exact=True)
    bound(receipt_path, receipt_hash)
    bound(prediction_ref['path'], prediction_ref['sha256'])
    for key in ('snapshot', 'sample', 'attachment', 'support_segments'):
        bound(receipt[key]['path'], receipt[key]['sha256'])
    return output

def plot_reference(ax, frame, *, label='实测参考（可靠可对齐范围）', color='#343B43', linewidth=.85, markersize=1.1):
    artists = []
    for run in continuous_reference_runs(frame):
        lines = ax.plot(frame.path_mm.iloc[run], frame.reference_power_cut_w.iloc[run],
                        color=color, lw=linewidth, marker='.', ms=markersize,
                        label=label if not artists else '_nolegend_')
        artists.extend(lines)
    return artists

def plot_frozen_prediction(ax, frame, *, sequence_id=None, label='预测（冻结旧模型）', color='#CE7139'):
    """Use prediction's own availability/sequence; never use reference masks."""
    valid = frame.prediction_available.to_numpy(bool) & np.isfinite(frame.cutting_power_w.to_numpy(float))
    x = frame.path_mm.to_numpy(float)
    sequence = frame.execution_instance.to_numpy() if sequence_id is None else np.asarray(sequence_id)
    if sequence.shape != (len(frame),):
        raise ValueError('Prediction sequence must cover the complete grid')
    same = (np.diff(x) > 0) & (sequence[1:] == sequence[:-1])
    if 'grid_index' in frame:
        same &= np.diff(frame.grid_index.to_numpy()) == 1
    starts = np.flatnonzero(valid & ~np.r_[False, valid[:-1] & same])
    ends = np.flatnonzero(valid & ~np.r_[valid[1:] & same, False])
    artists = []
    for a, b in zip(starts, ends):
        artists.extend(ax.plot(x[a:b+1], frame.cutting_power_w.iloc[a:b+1], color=color,
                               lw=.8, marker='.', ms=1.1, label=label if not artists else '_nolegend_'))
    return artists
