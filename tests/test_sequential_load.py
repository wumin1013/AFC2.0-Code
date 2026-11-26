"""Synthetic service-boundary tests; never research accuracy evidence."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
import pytest
import torch
from verify_unified_load_prediction import synthetic_records, SMALL_CANDIDATE
from spindle_power.load_prediction.common import save_json, file_hash
from spindle_power.load_prediction.models import ModelConfig
from spindle_power.load_prediction.training import TrainingConfig, train_two_stage, _sample_weights
from spindle_power.load_prediction.training_service import NOMINAL_ROUTE, evaluate_records, compare_evaluation_artifacts
from spindle_power.load_prediction.sequential_service import SequentialService, reference
from spindle_power.load_prediction import sequential_service as seq

@pytest.fixture
def small():
    torch.set_num_threads(1)
    _, records = synthetic_records((1,), points=4)
    for sample in records:
        sample.frame['entry_angle_rad']=0.
        sample.frame['exit_angle_rad']=0.
    model=train_two_stage(records, model_config=ModelConfig(**{**SMALL_CANDIDATE['load_model'],'route':NOMINAL_ROUTE}),
        config=TrainingConfig(main_epochs=1,residual_epochs=1),execute=True)
    later=deepcopy(records[:1])
    later[0].metadata.update(record_id='O1',group_id='O1',round_id=None,dataset_role='sequential_event',actual_event_order=None,tool_session=None)
    return model,records,later

@pytest.fixture
def service(small,tmp_path,monkeypatch):
    model, records, later=small
    sealed=tmp_path/'sealed'
    sealed.mkdir()
    model.save(sealed/'load.pt')
    scoring=dict(scale_w=263.,high_threshold_w=747.,low_threshold_w=77.)
    seal=dict(schema='paper1.k0_seal.1',status='accepted',selected='C64',full_epochs=dict(main=1,residual=1),
        snapshot_id='initial',frozen_scoring=scoring,k0=dict(model_files={'load.pt':file_hash(sealed/'load.pt')}))
    save_json(sealed/'k0_seal.json',seal)
    save_json(sealed/'scoring_parameters.json',dict(parameters=scoring))
    case=dict(k0_seal=reference(sealed/'k0_seal.json'),initial_snapshot={'path':'initial','sha256':'initial'},
        training_rule=dict(capacity='C64',epochs=seal['full_epochs'],initialization='fresh',selection=False,
            early_stopping=False,device='cpu',cpu_threads=4,cpu_interop_threads=1),
        groups=[dict(group_id=f'O{i}',snapshot=dict(path=f'O{i}',sha256=f'O{i}')) for i in (1,2,3)])
    save_json(tmp_path/'case.json',case)
    result=SequentialService(tmp_path/'case.json',tmp_path/'result')
    def loader(ref,group_id=None):
        if group_id is not None:
            assert group_id == 'O1', 'future group accessed'
            return deepcopy(later),dict(snapshot_id='O1')
        # Full original grouping is exercised without reading real machining files.
        _,initial=synthetic_records(points=4)
        return initial,dict(snapshot_id='initial')
    monkeypatch.setattr(seq,'load_snapshot',loader)
    return result

def test_unscored_cannot_release_or_train(service):
    with pytest.raises(FileNotFoundError): service.release('O1')
    with pytest.raises(FileNotFoundError): service.train(['O1'])
    with pytest.raises(ValueError): service.train(['O2'])
    assert not list(service.output.rglob('load.pt'))

def test_score_seal_idempotent_release_and_k0_unchanged(service):
    before=file_hash(service.k0_path)
    score=service.score('O1')
    assert score['memory'] is False
    assert score['scoring_parameters']==service.scoring
    first=service.release('O1')
    path=service.output/'released/O1.json'
    stamp=path.stat().st_mtime_ns
    assert service.release('O1')==first and path.stat().st_mtime_ns==stamp
    assert service.score('O1')==score
    assert file_hash(service.k0_path)==before
    with pytest.raises(FileNotFoundError): service.score('O2')
    with pytest.raises(ValueError): service.release('O3')

def test_failure_retained_and_no_release(service,monkeypatch):
    def fail(*args,**kwargs): raise RuntimeError('synthetic prediction failure')
    monkeypatch.setattr(seq,'evaluate_records',fail)
    with pytest.raises(RuntimeError): service.score('O1')
    failure=list((service.output/'attempts').glob('*/failure.json'))
    assert len(failure)==1 and json.loads(failure[0].read_text())['release_permitted'] is False
    with pytest.raises(FileNotFoundError): service.release('O1')

def test_cold_memory_is_zero_and_target_does_not_update(service):
    service.score('O1')
    result=service.memory_sensitivity('O1')
    assert result['rebuilt_from']==[] and result['target_memory_updated'] is False
    points=pd.read_csv(service.output/'memory_sensitivity/O1/K0/target/record_0000.csv')
    assert (points.memory_w==0.).all()
    saved=json.loads((service.output/'memory_sensitivity/O1/K0/target/memory.json').read_text())
    assert saved['entries']=={}

def test_changed_data_cannot_resume(service):
    service.score('O1'); service.release('O1')
    save_json(service.output/'runs/K1/identity.json',dict(data='different'))
    with pytest.raises(ValueError,match='恢复要求'): service.train(['O1'],resume=True)

def test_k0_modification_is_rejected(service):
    service.k0_path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='K0权重'): service.score('O1')


def test_actual_cumulative_training_and_two_history_rebuild(service,small,monkeypatch):
    _,_,later=small
    def loader(ref,group_id=None):
        if group_id is None:
            _,samples=synthetic_records(points=4)
            for sample in samples:
                sample.frame['entry_angle_rad']=0.
                sample.frame['exit_angle_rad']=0.
            return samples,dict(snapshot_id='initial')
        samples=deepcopy(later)
        samples[0].metadata.update(record_id=group_id,group_id=group_id)
        return samples,dict(snapshot_id=group_id)
    monkeypatch.setattr(seq,'load_snapshot',loader)
    before=file_hash(service.k0_path)
    service.score('O1'); service.release('O1')
    k1=service.train(['O1'])
    assert k1['status']=='sealed'
    assert service.train(['O1'])==k1
    service.score('O2'); service.release('O2')
    k2=service.train(['O1','O2'])
    assert k1['data_identity']!=k2['data_identity'] and k1['model_id']!=k2['model_id']
    service.score('O3')
    sensitivity=service.memory_sensitivity('O3')
    assert sensitivity['rebuilt_from']==['O1','O2'] and not sensitivity['target_memory_updated']
    before_target=json.loads((service.output/'memory_sensitivity/O3/K2/memory_before_target.json').read_text())
    assert before_target['entries'] and all(v['count']==2 and v['last_round']=='O2' for v in before_target['entries'].values())
    assert file_hash(service.k0_path)==before
    for name in ('K1','K2'):
        state=json.loads((service.output/f'runs/{name}/state.json').read_text())
        assert state['status']=='succeeded'

def test_single_event_equal_to_entire_five_category_round(small):
    _,initial,later=small
    weights=_sample_weights(initial+later)
    np.testing.assert_allclose(weights[:5],[.1]*5)
    assert weights[5]==.5
    assert later[0].metadata['round_id'] is None

def test_future_labels_masks_and_scales_do_not_change_predictions(small,tmp_path):
    model,_,later=small
    fixed=dict(scale_w=263.,high_threshold_w=747.,low_threshold_w=77.)
    a=evaluate_records(model,later,metric_parameters=fixed,output_dir=tmp_path/'a')
    changed=deepcopy(later)
    changed[0].frame['power_cut_w']=-999999.
    changed[0].frame['valid_residual']=False
    b=evaluate_records(model,changed,metric_parameters={**fixed,'scale_w':2.},output_dir=tmp_path/'b')
    assert a['records'][0]['prediction_fingerprint']==b['records'][0]['prediction_fingerprint']
    assert b['records'][0]['scored_points']==0 and b['records'][0]['predicted_points']==4
    columns=['line','execution_instance','segment_mm','path_mm','cutting_power_w','q05_w','q95_w']
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path/'a/record_0000.csv')[columns],pd.read_csv(tmp_path/'b/record_0000.csv')[columns])

def test_comparison_rejects_different_scoring_and_preserves_loss(small,tmp_path):
    model,_,later=small
    p=dict(scale_w=263.,high_threshold_w=747.,low_threshold_w=77.)
    evaluate_records(model,later,metric_parameters=p,output_dir=tmp_path/'a')
    evaluate_records(model,later,metric_parameters={**p,'scale_w':2.},output_dir=tmp_path/'b')
    with pytest.raises(ValueError,match='评分参数'): compare_evaluation_artifacts(dict(a=tmp_path/'a',b=tmp_path/'b'))
    masked=deepcopy(later)
    masked[0].frame.loc[0,'valid_residual']=False
    evaluate_records(model,masked,metric_parameters=p,output_dir=tmp_path/'c')
    result=compare_evaluation_artifacts(dict(a=tmp_path/'a',c=tmp_path/'c'))
    assert result['coverage'][0]['excluded_from_native_for_common']==1
    assert result['coverage'][0]['native_scored_points']==4
