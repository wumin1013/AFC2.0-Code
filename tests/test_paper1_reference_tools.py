"""Portable guards for the verification-layer attachment generator and display."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).parents[1] / 'scripts/verification'
sys.path.insert(0, str(SCRIPTS))
import generate_paper1_reference_attachments as generator
import paper1_reference_attachment as reference
from paper1_plot_reference_adapter import attach_reference, plot_reference, plot_frozen_prediction


def ref(path):
    return dict(path=str(path), sha256=reference.sha(path))


@pytest.fixture
def evidence(tmp_path):
    parent = pd.DataFrame(dict(line=1, execution_instance=0, path_mm=np.arange(7.),
        segment_mm=np.arange(7.), power_cut_w=[10., 11., 12., np.nan, 14., 15., 16.],
        valid_residual=[True, True, False, False, True, True, True]))
    prediction = parent.copy()
    prediction['scored'] = parent.valid_residual
    prediction['prediction_available'] = True
    prediction['cutting_power_w'] = np.arange(7.) + 10.
    attachment = parent[list(reference.KEYS)].copy()
    attachment['grid_index'] = np.arange(7)
    attachment['reference_power_cut_w'] = parent.power_cut_w
    attachment['reference_aligned'] = parent.power_cut_w.notna()
    attachment['reference_support_segment_id'] = [0, 0, 1, -1, 2, 3, 3]
    attachment['reference_alignment_error_mm'] = 0.
    attachment['reference_kind'] = 'postoperative_measured_idle'
    attachment['reference_evidence_grade'] = 'observed_platform_with_background_assumption'
    files = dict(snapshot=tmp_path/'snapshot.json', sample=tmp_path/'sample.csv',
                 prediction=tmp_path/'prediction.csv', attachment=tmp_path/'reference.csv',
                 support_segments=tmp_path/'support.json')
    files['snapshot'].write_text('{}', encoding='utf8')
    files['support_segments'].write_text('{}', encoding='utf8')
    for key, frame in (('sample', parent), ('prediction', prediction), ('attachment', attachment)):
        frame.to_csv(files[key], index=False, encoding='utf8')
    receipt = dict(status='candidate_verified_not_formally_released', **{key: ref(path) for key, path in files.items()})
    receipt_path = tmp_path/'receipt.json'
    reference.write_json(receipt_path, receipt)
    return prediction, attachment, receipt, receipt_path, files


def attach(data, **kwargs):
    prediction, _, _, receipt_path, files = data
    return attach_reference(prediction, receipt_path, snapshot_path=files['snapshot'], sample_path=files['sample'], **kwargs)


def test_legacy_api_preserves_every_target_prediction_and_score(evidence):
    joined = attach(evidence)
    pd.testing.assert_frame_equal(joined[evidence[0].columns], evidence[0], check_exact=True)
    assert joined.loc[2, 'reference_aligned'] and not joined.loc[2, 'scored']
    assert joined.loc[3, 'prediction_available'] and not joined.loc[3, 'reference_aligned']


def test_development_attachment_needs_explicit_bound_prediction(evidence):
    _, _, receipt, path, files = evidence
    receipt['prediction'] = None
    reference.write_json(path, receipt)
    with pytest.raises(ValueError, match='explicit frozen prediction'):
        attach(evidence)
    joined = attach(evidence, prediction_reference=ref(files['prediction']))
    pd.testing.assert_frame_equal(joined[evidence[0].columns], evidence[0], check_exact=True)


def test_legacy_bound_prediction_cannot_be_silently_replaced(evidence):
    with pytest.raises(ValueError, match='Cannot replace'):
        attach(evidence, prediction_reference=dict(path='other.csv', sha256='other'))


@pytest.mark.parametrize('change', ['scored', 'target', 'path', 'reference', 'attachment_hash'])
def test_even_new_hashes_cannot_redefine_parent_scoring_or_reference(evidence, change):
    prediction, attachment, receipt, path, files = evidence
    if change in ('scored', 'target', 'path'):
        column = dict(scored='scored', target='power_cut_w', path='path_mm')[change]
        prediction.loc[0, column] = False if change == 'scored' else 999.
        prediction.to_csv(files['prediction'], index=False, encoding='utf8')
        receipt['prediction'] = ref(files['prediction'])
    elif change == 'reference':
        attachment.loc[0, 'reference_power_cut_w'] += 1.
        attachment.to_csv(files['attachment'], index=False, encoding='utf8')
        receipt['attachment'] = ref(files['attachment'])
    else:
        files['attachment'].write_bytes(files['attachment'].read_bytes() + b'\n')
    reference.write_json(path, receipt)
    with pytest.raises((ValueError, AssertionError)):
        attach(evidence)


def test_reference_lines_keep_support_gaps_and_singletons(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    joined = attach(evidence)
    fig, ax = plt.subplots()
    artists = plot_reference(ax, joined)
    assert [len(line.get_xdata()) for line in artists] == [2, 1, 1, 2]
    assert all(line.get_marker() == '.' for line in artists)
    predicted = plot_frozen_prediction(ax, joined, sequence_id=np.zeros(len(joined), int))
    assert [len(line.get_xdata()) for line in predicted] == [7]
    plt.close(fig)


def test_prediction_does_not_bridge_a_missing_original_grid_index(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    joined = attach(evidence).iloc[[0, 2]].copy()
    fig, ax = plt.subplots()
    artists = plot_frozen_prediction(ax, joined)
    assert [len(line.get_xdata()) for line in artists] == [1, 1]
    plt.close(fig)


def test_alignment_source_is_explicit_hash_bound_and_requires_optional_api(tmp_path):
    path = tmp_path/'alignment.py'
    path.write_text('def align_spatial_channels(*args, **kwargs): pass\n', encoding='utf8')
    with pytest.raises(ValueError, match='support_provenance API'):
        generator.checked_alignment(ref(path))
    path.write_text('def align_spatial_channels(*args, support_provenance=False): return support_provenance\n', encoding='utf8')
    binding = ref(path)
    function = generator.checked_alignment(binding)
    assert function() is False and function(support_provenance=True) is True
    path.write_text('changed', encoding='utf8')
    with pytest.raises(ValueError, match='Artifact hash changed'):
        generator.checked_alignment(binding)


def development_snapshot():
    return dict(records=[dict(kind='load', metadata=dict(record_id=f'R{i}', dataset_role='development')) for i in range(450)])


@pytest.mark.parametrize('change', ['partial', 'duplicate', 'event'])
def test_development_scope_refuses_partial_duplicate_or_event_records(change):
    snapshot = development_snapshot()
    if change == 'partial':
        snapshot['records'].pop()
    elif change == 'duplicate':
        snapshot['records'][-1]['metadata']['record_id'] = 'R0'
    else:
        snapshot['records'][-1]['metadata']['group_id'] = 'O3'
    with pytest.raises(ValueError):
        generator.development_records(snapshot)


def test_development_scope_retains_all_records_and_separate_preflight_outputs(tmp_path):
    assert len(generator.development_records(development_snapshot())) == 450
    assert generator.record_output(tmp_path, 'R1', preflight=True) == tmp_path/'preflight/R1'
    assert generator.record_output(tmp_path, 'R1') == tmp_path/'artifacts/R1'
    with pytest.raises(ValueError):
        generator.record_output(tmp_path, '../R1')
