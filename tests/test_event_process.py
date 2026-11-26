"""Later process geometry reuses NC feed and retains unknown source rows."""
from pathlib import Path
from types import SimpleNamespace
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from spindle_power.load_prediction.event_process import prepare_event_process

def test_original_process_feed_cannot_change_steady_decisions(tmp_path):
    plan=SimpleNamespace(frame=pd.DataFrame(dict(line=[2,2],execution_instance=[0,0],
        block_length_mm=[10.,10.],segment_mm=[.1,10.],path_mm=[.1,10.],feed_program_mm_min=[4000.,4000.])))
    def source(name,feed):
        path=tmp_path/name
        lines=['序号 N ap ae F MRR s(mm) S G']
        lines += [f'{i} 2 2 3 {feed} {feed*100} {i/100} 10000 G1X10' for i in range(1,101)]
        lines.append(f'101 999 2 3 {feed} {feed*100} 9 10000 G1X99')
        path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return path
    a,ea=prepare_event_process(source('a.txt',30.),plan)
    b,eb=prepare_event_process(source('b.txt',9000.),plan)
    assert len(a)==101 and a.source_point_index.tolist()==list(range(1,102))
    assert a.source_file_line.is_unique
    assert not a.iloc[-1].steady and a.iloc[-1].segmentation_run==-1
    assert a.iloc[-1].mapping_status=='unmapped_nc_group'
    assert (a.iloc[:100].feed_program_mm_min==4000.).all()
    assert ea['source_points_preserved'] and eb['source_points_preserved']
    assert a.steady.tolist()==b.steady.tolist()
    assert a.steady.any(), 'A constant, sufficiently long cutting block must retain steady candidates'
    np.testing.assert_allclose(a.iloc[:100].physical_path_end_mm,np.arange(1,101)*.1)
