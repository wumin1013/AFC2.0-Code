"""Python result exports from sealed artifacts; no training or data release."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import file_hash,save_json

METHODS = dict(fixed_physical='固定物理基线',physical_reused_main='复用主趋势',main_only='复用主趋势',
               nominal_no_memory='名义输入、无记忆',without_memory='名义输入、无记忆',
               with_memory='位置记忆（敏感性）',measured_feed='实测进给（诊断）')


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.unicode_minus':False,'legend.frameon':False})
    return plt


def _save_figure(fig,path):
    for suffix in ('pdf','svg','png'):
        fig.savefig(path.with_suffix('.'+suffix),dpi=300,bbox_inches='tight')


def export_initial(seal_dir, output_dir, *, plot=True):
    source,output=Path(seal_dir),Path(output_dir)
    seal=_read(source/'k0_seal.json')
    if seal.get('status')!='accepted': raise ValueError('初始结果尚未验收')
    hashes=_read(source/'frozen_exports/sha256.json')
    fold_path=source/'frozen_exports/folds.csv'
    if file_hash(fold_path)!=hashes['folds.csv']: raise ValueError('冻结逐折结果身份不一致')
    folds=pd.read_csv(fold_path,float_precision='round_trip')
    methods=list(folds.method.unique())
    # Preserve source method names if the experimental protocol uses aliases.
    labels={**METHODS,'nominal_main_only':'复用所训练主趋势','measured_feed_diagnostic':'实测进给（诊断）'}
    common=folds[folds.score_scope.eq('common')].copy()
    selected=common[common.candidate.eq(seal['selected'])].copy()
    if (selected.duplicated(['method','fold']).any() or
        any(set(rows.fold)!={f'F{i}' for i in range(1,10)} for _,rows in selected.groupby('method')) or len(selected)!=45):
        raise ValueError('初始五项九折结果不完整')
    output.mkdir(parents=True,exist_ok=False)
    common.to_csv(output/'initial_folds.csv',index=False,encoding='utf-8')
    metrics=['rmse_w','mae_w','high_load_underestimate_rate','interval_coverage']
    if 'interval_coverage' not in selected or selected.interval_coverage.isna().all():
        metrics[-1]='coverage'
    summary=[]
    for method,rows in selected.groupby('method',sort=False):
        row=dict(method=method,label=labels.get(method,method),folds=len(rows))
        for metric in metrics:
            values=rows[metric].dropna()
            row[metric+'_mean']=values.mean() if len(values) else None
            row[metric+'_sd']=values.std(ddof=1) if len(values)>1 else None
        summary.append(row)
    pd.DataFrame(summary).to_csv(output/'initial_summary.csv',index=False,encoding='utf-8')
    def val(row,key,percent=False):
        mean,sd=row[key+'_mean'],row[key+'_sd']
        if mean is None: return '--'
        factor=100 if percent else 1
        return f'{mean*factor:.2f} $\\pm$ {sd*factor:.2f}'
    lines=[r'\begin{table}[htbp]',r'\centering',r'\scriptsize',r'\setlength{\tabcolsep}{3pt}',
        r'\caption{初始九折开发比较。数值为折均值及样本标准差；高负载低估率和区间覆盖率以百分数表示。}',
        r'\label{tab:initial-results}',r'\begin{tabularx}{\linewidth}{@{}Xrrrr@{}}',r'\toprule',r'方法 & RMSE (W) & MAE (W) & 高负载低估率 & 区间覆盖率 \\',r'\midrule']
    for row in summary:
        lines.append(' & '.join([row['label'],val(row,'rmse_w'),val(row,'mae_w'),val(row,'high_load_underestimate_rate',True),val(row,metrics[-1],True)])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabularx}',r'\end{table}'])
    (output/'initial_results.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if plot:
        plt=_plotting()
        fig,axes=plt.subplots(1,2,figsize=(4.8,2.8),sharey=True,gridspec_kw={'width_ratios':[1.2,1]})
        colors=['#80858b','#aaaeb2','#356d96','#6f9d91','#ac899f']
        for i,method in enumerate(methods):
            rows=selected[selected.method.eq(method)].sort_values('fold')
            for ax,key,factor in zip(axes,['rmse_w','high_load_underestimate_rate'],[1,100]):
                values=rows[key].to_numpy()*factor
                ax.scatter(values,i+np.linspace(-.12,.12,len(values)),s=13,color=colors[i],alpha=.75,zorder=3)
                ax.plot(np.mean(values),i,marker='D',color='black',markersize=4,zorder=4)
                ax.axhline(i,color='#ececec',linewidth=.6,zorder=0)
        axes[0].set_yticks(range(len(methods)),[labels.get(m,m) for m in methods])
        axes[0].invert_yaxis()
        for i,ax in enumerate(axes):
            ax.set_title(chr(97+i),loc='left',fontweight='bold')
            ax.set_xlim(left=0)
        axes[0].set_xlabel('RMSE (W)')
        axes[0].set_xticks([0,100,200,300])
        axes[1].set_xlabel('高负载低估率 (%)'); axes[1].set_xlim(0,100)
        fig.tight_layout(w_pad=1.5)
        _save_figure(fig,output/'initial_comparison'); plt.close(fig)
    contract=dict(claim='Compare initial model components while retaining high-load underestimation.',
        evidence='All 9 held-out fold summaries, selected C64; internal development, not independent subsequent tests',
        n='9 folds, one fixed seed, no inferential p-values',center='diamond: arithmetic fold mean',
        spread='individual fold points; table sample SD is not a confidence interval',
        style='Original quantitative comparison with direct labels; no external figure copied or claimed as a verified precedent',
        source=dict(path=str(fold_path),sha256=file_hash(fold_path)),raw_data_modified=False,
        implementation=dict(path=__file__,sha256=file_hash(__file__)))
    save_json(output/'export.json',dict(contract=contract,artifacts={p.name:file_hash(p) for p in output.iterdir() if p.is_file()}))
    return dict(output=str(output),folds=len(selected),methods=methods)


def export_sequential(service_dir, output_dir, *, plot=True):
    source,output=Path(service_dir),Path(output_dir)
    rows,native_rows,paired,coverage=[],[],[],[]
    for group in ('O1','O2','O3'):
        directory=source/'scores'/group
        if not (directory/'receipt.json').is_file(): continue
        receipt=_read(directory/'receipt.json')
        if receipt['status']!='sealed': raise ValueError('序贯评分未封存')
        for item in receipt['artifacts'].values():
            if file_hash(item['path'])!=item['sha256']: raise ValueError('序贯逐点证据已改变')
        native={}
        for model in receipt['models']:
            report=_read(directory/model/'evaluation.json')
            native[model]=report
            for record in report['records']:
                native_rows.append(dict(group_id=group,model=model,record_id=record['record_id'],score_scope='native',**record['metrics']))
        if (directory/'common/comparison.json').exists():
            common=_read(directory/'common/comparison.json')
            for model,report in common['comparisons'].items():
                for record in report['records']:
                    rows.append(dict(group_id=group,model=model,record_id=record['record_id'],score_scope='common',**record['metrics']))
                    coverage.append(dict(group_id=group,model=model,record_id=record['record_id'],**record['coverage']))
            for item in common['paired_differences']:
                paired.append(dict(group_id=group,left=item['left'],right=item['right'],record_id=item['record_id'],**item['deltas']))
        else:
            if group!='O1' or set(native)!={'K0'}:
                raise ValueError('配对事件缺少共同评分，不能用原始范围代替')
            for record in native['K0']['records']:
                rows.append(dict(group_id=group,model='K0',record_id=record['record_id'],score_scope='single_model',**record['metrics']))
                coverage.append(dict(group_id=group,model='K0',record_id=record['record_id'],**record['coverage']))
    failures=[dict(path=str(p),**_read(p)) for root in ('attempts','scores','runs')
              for p in sorted((source/root).glob('*/failure.json'))]
    failures.extend(dict(path=str(p),**_read(p)) for p in sorted((source/'runs').glob('*/state.json')) if _read(p).get('status')=='failed')
    output.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'sequential_metrics.csv',index=False,encoding='utf-8')
    pd.DataFrame(native_rows).to_csv(output/'native_metrics.csv',index=False,encoding='utf-8')
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False,encoding='utf-8')
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False,encoding='utf-8')
    if plot and len(frame):
        plt=_plotting()
        fig,axes=plt.subplots(1,2,figsize=(4.8,2.7))
        for ax,key,label in zip(axes,['rmse_w','mae_w'],['RMSE (W)','MAE (W)']):
            for model,part in frame.groupby('model',sort=True):
                x=part.group_id.str.removeprefix('O').astype(int)
                ax.plot(x,part[key],'-o' if model=='K0' else 's',label=model,markersize=4)
            ax.set_xticks([1,2,3],['O1','O2','O3']); ax.set_xlabel('声明回放事件'); ax.set_ylabel(label)
            ax.legend()
        fig.tight_layout(); _save_figure(fig,output/'update_curve'); plt.close(fig)
    save_json(output/'export.json',dict(scored_groups=sorted({r['group_id'] for r in rows}),failures=failures,
        complete=bool(rows) and set(r['group_id'] for r in rows)=={'O1','O2','O3'},
        delta_sign='updated_minus_K0',curve_score_scope='paired events use common positions; O1 uses K0 native positions',
        artifacts={p.name:file_hash(p) for p in output.iterdir() if p.is_file()},
        interpretation='sequential prospective replay; cumulative retraining is not rapid-transfer validation'))
    return dict(output=str(output),records=len(rows),failures=len(failures))
