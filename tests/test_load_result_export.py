"""Export integrity and common-scope regressions, using synthetic metrics."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import pytest
from spindle_power.load_prediction.common import file_hash,save_json
from spindle_power.load_prediction.result_export import export_initial,export_sequential

def test_initial_requires_unique_folds_and_verified_data(tmp_path):
    source=tmp_path/'seal'
    rows=[dict(candidate='C64',method=m,fold=f'F{i}',score_scope='common',rmse_w=float(i),
        mae_w=float(i)/2,high_load_underestimate_rate=.2,interval_coverage=.9)
        for m in ('fixed_physical','physical_reused_main','without_memory','with_memory','measured_feed_diagnostic')
        for i in range(1,10)]
    source.mkdir(); (source/'frozen_exports').mkdir()
    file=source/'frozen_exports/folds.csv'
    pd.DataFrame(rows).to_csv(file,index=False)
    save_json(source/'k0_seal.json',dict(status='accepted',selected='C64'))
    save_json(source/'frozen_exports/sha256.json',{'folds.csv':file_hash(file)})
    export_initial(source,tmp_path/'good',plot=False)
    summary=pd.read_csv(tmp_path/'good/initial_summary.csv')
    assert (summary.rmse_w_mean==5.).all()
    assert summary.loc[summary.method.eq('physical_reused_main'),'label'].item()=='复用主趋势'
    with file.open('a') as stream: stream.write('\n')
    with pytest.raises(ValueError,match='身份'):
        export_initial(source,tmp_path/'bad',plot=False)
    rows[1]['fold']='F1'
    pd.DataFrame(rows).to_csv(file,index=False)
    save_json(source/'frozen_exports/sha256.json',{'folds.csv':file_hash(file)})
    with pytest.raises(ValueError,match='不完整'):
        export_initial(source,tmp_path/'duplicate',plot=False)

def test_sequential_uses_common_and_retains_native_and_failures(tmp_path):
    source=tmp_path/'service'; score=source/'scores/O2'
    record=lambda rmse:dict(record_id='O2',metrics=dict(rmse_w=rmse,mae_w=rmse/2),coverage=dict(native_scored_points=3))
    for model,rmse in [('K0',9.),('K1',8.)]:
        save_json(score/model/'evaluation.json',dict(records=[record(rmse)]))
    save_json(score/'common/comparison.json',dict(
        comparisons={'K0':dict(records=[record(5.)]),'K1':dict(records=[record(6.)])},
        paired_differences=[dict(left='K0',right='K1',record_id='O2',deltas=dict(rmse_w=1.))]))
    save_json(score/'receipt.json',dict(status='sealed',models={'K0':{},'K1':{}},
        artifacts={p.relative_to(score).as_posix():dict(path=str(p),sha256=file_hash(p))
            for p in score.rglob('*') if p.is_file()}))
    save_json(source/'attempts/failed/failure.json',dict(status='failed',error='retained'))
    export_sequential(source,tmp_path/'out',plot=False)
    assert pd.read_csv(tmp_path/'out/sequential_metrics.csv').rmse_w.tolist()==[5.,6.]
    assert pd.read_csv(tmp_path/'out/native_metrics.csv').rmse_w.tolist()==[9.,8.]
    assert pd.read_csv(tmp_path/'out/paired_differences.csv').rmse_w.item()==1.
    assert json.loads((tmp_path/'out/export.json').read_text())['failures'][0]['error']=='retained'

def test_empty_export_does_not_claim_case_completed(tmp_path):
    export_sequential(tmp_path/'absent',tmp_path/'out',plot=False)
    assert not json.loads((tmp_path/'out/export.json').read_text())['complete']
    assert not (tmp_path/'out/update_curve.png').exists()
