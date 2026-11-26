"""Map a later program's process points and reuse the frozen steady segmenter."""
import numpy as np
import pandas as pd
from .samples import read_process_table
from interval_analysis.segmentation import SegmentationConfig, SegmentationPipeline


def prepare_event_process(path, plan, *, line_offset=0):
    """Retain every source row; F/MRR for segmentation come from the NC."""
    raw = read_process_table(path, plan, line_offset=line_offset).copy()
    if 'source_point_index' not in raw or raw.source_file_line.duplicated().any():
        raise ValueError('后续工艺必须保留唯一原始点身份')
    raw['source_index'] = np.arange(len(raw))
    raw['feed_program_mm_min'] = np.nan
    for (line, instance), rows in raw.loc[raw.execution_instance >= 0].groupby(['line','execution_instance'],sort=False):
        grid = plan.frame.loc[(plan.frame.line == line)&(plan.frame.execution_instance == instance)]
        values = grid.feed_program_mm_min.unique()
        if len(values) == 1 and np.isfinite(values[0]) and values[0] > 0:
            raw.loc[rows.index,'feed_program_mm_min'] = values[0]
    raw['physical_path_start_mm'] = raw.nc_path_mm - raw.segment_mm + raw.segment_start_mm
    raw['physical_path_end_mm'] = raw.nc_path_mm
    raw['steady'] = False
    config = SegmentationConfig()
    valid = (raw.mapping_status.eq('mapped_uniform_nc') & np.isfinite(raw.feed_program_mm_min)
             & np.isfinite(raw.physical_path_start_mm) & np.isfinite(raw.physical_path_end_mm))
    run, ids = -1, np.full(len(raw),-1,dtype=int)
    for i in range(len(raw)):
        if not valid.iloc[i]:
            continue
        if not (i>0 and valid.iloc[i-1] and abs(raw.physical_path_start_mm.iloc[i]-raw.physical_path_end_mm.iloc[i-1]) <= config.path_tolerance_mm):
            run += 1
        ids[i] = run
    raw['segmentation_run'] = ids
    evidence = []
    for run, rows in raw.loc[ids>=0].groupby('segmentation_run',sort=False):
        inputs=pd.DataFrame(dict(source_index=rows.source_index,path_start=rows.physical_path_start_mm,
            path_end=rows.physical_path_end_mm,line_id=rows.line,line_no_raw=rows.source_nc_line,
            ap=rows.ap_mm,ae=rows.ae_mm,F_program=rows.feed_program_mm_min,
            path_source='author_confirmed_uniform_nc_block_points',path_is_physical=True))
        result=SegmentationPipeline(config).run(inputs)
        d=result.diagnostics
        path_info=d.get('path',{})
        structural=bool(d.get('postprocess_validation_passed') and d.get('strict_transition_semantics_valid')
            and path_info.get('is_physical') and not path_info.get('used_nonphysical_fallback')
            and (not d.get('fallback_used') or (d.get('fallback_scope')=='local_verified' and d.get('fallback_validated'))))
        detail=result.point_labels[['source_index','interval_id','segment_type','review_required']].merge(
            result.intervals[['interval_id','confidence_level','review_required']].rename(columns={'review_required':'interval_review_required'}),
            on='interval_id',validate='many_to_one')
        if set(detail.source_index)!=set(rows.source_index) or detail.source_index.duplicated().any():
            raise ValueError('分段丢失或重复工艺源点')
        accepted=detail.loc[detail.segment_type.eq('steady') & detail.confidence_level.eq('high')
            & detail.review_required.eq(False) & detail.interval_review_required.eq(False),'source_index']
        if structural:
            raw.loc[raw.source_index.isin(accepted),'steady']=True
        evidence.append(dict(run=int(run),structurally_valid=structural,diagnostics=d,
            point_labels=result.point_labels.to_dict('records'),intervals=result.intervals.to_dict('records')))
    return raw,dict(config=config.to_dict(),runs=evidence,source_rows=len(raw),
        steady_points=int(raw.steady.sum()),source_points_preserved=True,
        feed_policy='NC programmed F; original process F/MRR retained only as source evidence')
