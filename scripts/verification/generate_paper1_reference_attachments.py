"""Explicit-input reference candidates: fixed six examples or all 450 development records.

The explicit alignment source must expose the audited optional support API.
No production source is changed by this script and no model is trained. --preflight-only validates
one explicitly selected development record in memory before a separate full run.
"""
from pathlib import Path
import argparse
import copy
from importlib import import_module
import inspect
import os
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from paper1_reference_attachment import (KEYS, bound, build_reference_attachment, continuous_reference_runs,
                                  load_overlay, read_json, sha, write_json)

HERE = Path(__file__).resolve().parent
FIXED_IDS = ['R81_外形', 'R29_台阶', 'R10_斜面', 'R07_曲面', 'R37_混合', 'O1']


def development_records(snapshot):
    """Reject partial or sequential-event snapshots before resolving raw files."""
    records = [row for row in snapshot['records'] if row['kind'] == 'load']
    ids = [row['metadata']['record_id'] for row in records]
    if len(records) != 450 or len(set(ids)) != 450:
        raise ValueError('The complete unique 450-record development snapshot is required')
    for row in records:
        metadata = row['metadata']
        if (metadata.get('dataset_role') != 'development' or metadata.get('group_id') is not None
                or metadata['record_id'] in ('O1', 'O2', 'O3')):
            raise ValueError('Only development records are allowed; sequential events are excluded')
    return records


def record_output(root, record_id, *, preflight=False):
    if not record_id or record_id in ('.', '..') or '/' in record_id or '\\' in record_id:
        raise ValueError('Unsafe record identity')
    return Path(root) / ('preflight' if preflight else 'artifacts') / record_id


def checked_alignment(reference, package='spindle_power.load_prediction'):
    """Fail before raw data is read if the optional formal API is not installed."""
    path = bound(reference['path'], reference['sha256'])
    module = load_overlay(path, package)
    if 'support_provenance' not in inspect.signature(module.align_spatial_channels).parameters:
        raise ValueError('The explicit alignment source lacks the audited optional support_provenance API')
    bound(path, reference['sha256'])
    return module.align_spatial_channels


def producer_package(producer):
    producer = producer.resolve()
    source_root = producer / 'src'
    for name in ('spindle_power.load_prediction', 'load_prediction', 'project.load_prediction'):
        directory = source_root.joinpath(*name.split('.'))
        if (directory / '__init__.py').is_file():
            directory = directory.resolve()
            if not directory.is_relative_to(producer):
                raise ValueError(f'Producer package leaves its recorded root: {directory}')
            sys.path.insert(0, str(source_root))
            return name, directory
    raise FileNotFoundError(f'Cannot locate the load-prediction package in {source_root}')


def worker(task_path):
    started = time.perf_counter()
    task = read_json(task_path)
    bound(Path(__file__), task['generator_sha256'])
    for name, digest in task['candidate_source_sha256'].items():
        bound(task['alignment_source']['path'] if name == 'alignment_source' else HERE / name, digest)
    output_root = Path(task['output_root'])
    preflight = task['preflight_only']
    producer = Path(task['producer_root'])
    package, producer_directory = producer_package(producer)
    feed_samples = import_module(f'{package}.feed_samples')
    measurement_module = import_module(f'{package}.measurement')
    artifact_paths = import_module(f'{package}.artifact_paths')
    for module in (feed_samples, measurement_module, artifact_paths):
        if Path(module.__file__).resolve().parent != producer_directory:
            raise ValueError(f'Wrong producer source loaded for {module.__name__}')
    reference_align = checked_alignment(task['alignment_source'], package)
    snapshot_path = bound(task['snapshot']['path'], task['snapshot']['sha256'])
    sample_path = bound(task['sample']['path'], task['sample']['sha256'])
    prediction_path = bound(task['prediction']['path'], task['prediction']['sha256']) if task['prediction'] else None
    snapshot = read_json(snapshot_path)
    metadata = task['metadata']
    if task['mode'] == 'development':
        entries = development_records(snapshot)
    elif task['mode'] == 'fixed6' and metadata['record_id'] in FIXED_IDS:
        entries = [row for row in snapshot['records'] if row['kind'] == 'load']
    else:
        raise ValueError('Unregistered candidate scope')
    entry = next(row for row in entries if row['metadata']['record_id'] == metadata['record_id'])
    assert entry['metadata'] == metadata and entry['sha256'] == task['sample']['sha256']
    output = record_output(output_root, metadata['record_id'], preflight=preflight)
    output.mkdir(parents=True, exist_ok=False)
    evidence = copy.deepcopy(snapshot['source_snapshot']['base_validation'])
    for key in ('report_path', 'model_path'):
        if key in evidence:
            p = Path(evidence[key])
            evidence[key] = str(p if p.is_absolute() else snapshot_path.parent / p)
    report = read_json(bound(evidence['report_path'], evidence['report_sha256']))
    if 'model_path' in evidence:
        bound(evidence['model_path'], evidence['model_sha256'])
    for name, expected in report['production_source_sha256'].items():
        bound(producer_directory / name, expected)
    relocation_payload = read_json(bound(task['relocations']['path'], task['relocations']['sha256']))
    relocations = relocation_payload.get('artifact_relocations', relocation_payload)
    resolver = artifact_paths.ArtifactResolver(relocations)
    measurement_path = bound(resolver.resolve(metadata['resolved_files']['measurement']), metadata['file_sha256']['measurement'])
    measurement = measurement_module.read_measurement(measurement_path, sample_period_s=metadata.get('sample_period_s'))
    if evidence['mode'] == 'hybrid_estimated_labels':
        labels = import_module(f'{package}.hybrid_labels')
        base_power = import_module(f'{package}.base_power')
        base = labels.prepare_hybrid_power(
            measurement, base_power.BasePowerModel.load(evidence['model_path']), evidence, metadata)
        original_label = metadata['hybrid_label']
        generated_label = base['hybrid_label_metadata']
    elif evidence['mode'] == 'record_idle_labels':
        labels = import_module(f'{package}.record_idle')
        base = labels.prepare_record_idle_power(measurement, evidence, metadata)
        original_label = metadata['record_idle_label']
        generated_label = base['label_metadata']
    else:
        raise ValueError('Unregistered reference mode')
    if generated_label != original_label:
        write_json(output / 'label_lineage_difference.json', {
            key: {'parent': original_label.get(key), 'recomputed': generated_label.get(key)}
            for key in set(generated_label) | set(original_label)
            if original_label.get(key) != generated_label.get(key)})
    lineage_only = {'mask_sha256', 'parent_computation', 'report_sha256'} if evidence['mode'] == 'hybrid_estimated_labels' else set()
    for key in (set(original_label) | set(generated_label)) - lineage_only:
        assert generated_label.get(key) == original_label.get(key), key
    for key in ('valid_sample_ranges', 'valid_samples', 'raw_sample_count'):
        if key in original_label:
            assert generated_label[key] == original_label[key], key
    expected_raw_valid = np.zeros(len(measurement.time_s), bool)
    for start, end in original_label['valid_sample_ranges']:
        expected_raw_valid |= (measurement.source_indices >= start) & (measurement.source_indices < end)
    np.testing.assert_array_equal(np.asarray(base['valid'], bool), expected_raw_valid)
    if original_label.get('idle_power_w') is not None:
        np.testing.assert_array_equal(np.asarray(base['power_w']), np.full(len(measurement.time_s), original_label['idle_power_w']))
    parent = pd.read_csv(sample_path, float_precision='round_trip')
    prediction = pd.read_csv(prediction_path, float_precision='round_trip') if prediction_path else None
    if prediction is not None:
        for key in KEYS:
            np.testing.assert_array_equal(parent[key], prediction[key])
    attachment, info, verification = build_reference_attachment(parent, measurement, base, metadata,
                                                                feed_samples.align_spatial_channels, reference_align)
    scored = parent.valid_residual.to_numpy(bool)
    if prediction is not None:
        np.testing.assert_array_equal(scored, prediction.scored.to_numpy(bool))
    available = attachment.reference_aligned.to_numpy(bool)
    geometry = np.isfinite(parent[['ap_mm', 'ae_mm']].to_numpy(float)).all(axis=1)
    np.testing.assert_array_equal(parent.loc[scored, 'power_cut_w'], attachment.loc[scored, 'reference_power_cut_w'])
    metrics = None
    if prediction is not None:
        error = prediction.loc[scored, 'cutting_power_w'].to_numpy(float) - parent.loc[scored, 'power_cut_w'].to_numpy(float)
        metrics = {'rmse_w': float(np.sqrt(np.mean(error ** 2))), 'mae_w': float(np.abs(error).mean())}
        for name, value in metrics.items():
            assert np.isclose(value, task['metrics'][name], rtol=1e-10, atol=0), (name, value)
    counts = {'grid_points': len(parent), 'scored_points': int(scored.sum()), 'reference_points': int(available.sum()),
              'display_recovered': int((available & ~scored).sum()),
              'display_recovered_missing_geometry': int((available & ~scored & ~geometry).sum()),
              'display_recovered_other_exclusion': int((available & ~scored & geometry).sum()),
              'reference_missing': int((~available).sum())}
    if metadata['record_id'] == 'O1':
        assert counts == {'grid_points': 3496, 'scored_points': 2213, 'reference_points': 2410,
                          'display_recovered': 197, 'display_recovered_missing_geometry': 43,
                          'display_recovered_other_exclusion': 154, 'reference_missing': 1086}, counts
    runs = continuous_reference_runs(attachment)
    assert sum(map(len, runs)) == counts['reference_points']
    attachment_ref, support_ref = None, None
    if not preflight:
        attachment_path = output / 'reference_attachment.csv'
        attachment.to_csv(attachment_path, index=False, encoding='utf-8')
        write_json(output / 'reference_support_segments.json', info)
        restored = pd.read_csv(attachment_path, float_precision='round_trip')
        pd.testing.assert_frame_equal(restored, attachment, check_exact=True)
        attachment_ref = {'path': str(attachment_path), 'sha256': sha(attachment_path)}
        support_ref = {'path': str(output / 'reference_support_segments.json'), 'sha256': sha(output / 'reference_support_segments.json')}
    for item in (task['sample'], task['prediction'], task['snapshot']):
        if item:
            bound(item['path'], item['sha256'])
    bound(measurement_path, metadata['file_sha256']['measurement'])
    bound(Path(__file__), task['generator_sha256'])
    for name, digest in task['candidate_source_sha256'].items():
        bound(task['alignment_source']['path'] if name == 'alignment_source' else HERE / name, digest)
    receipt = {'schema': 'afc.reference_display_candidate.2',
        'status': 'single_record_preflight_verified' if preflight else 'candidate_verified_not_formally_released',
        'record_id': metadata['record_id'], 'snapshot': task['snapshot'], 'sample': task['sample'],
        'prediction': task['prediction'], 'producer_root': str(producer),
        'generator_sha256': task['generator_sha256'], 'relocations': task['relocations'],
        'candidate_source_sha256': task['candidate_source_sha256'],
        'mode': task['mode'], 'preflight_only': preflight,
        'producer_report': {'path': evidence['report_path'], 'sha256': evidence['report_sha256']},
        'producer_sources': report['production_source_sha256'],
        'alignment_source': task['alignment_source'],
        'historical_alignment_sha256': sha(feed_samples.__file__),
        'base_mask_sha256': generated_label['mask_sha256'], 'raw_sample_count': len(measurement.time_s),
        'parent_label_mask_sha256': original_label['mask_sha256'],
        'base_mask_identity_note': 'Raw valid arrays and numerical label fields are exact; hybrid parent_computation and report lineage keep distinct identity hashes.',
        'label_metadata_exact_except_declared_lineage': True,
        'declared_label_lineage_exceptions': sorted(lineage_only),
        'reference_kind': original_label.get('reference_kind', 'postoperative_measured_idle'),
        'base_status': base.get('status'), 'formal_metrological_qualification': base.get('formal_metrological_qualification'),
        'base_warnings': base.get('warnings', []), 'base_validity_exact': True,
        'position_offset_mm': metadata['position_offset_mm'], 'counts': counts,
        'metrics_unchanged': metrics, 'verification': verification,
        'display_support_segments': len(runs), 'singleton_reference_segments': sum(len(r) == 1 for r in runs),
        'attachment': attachment_ref, 'support_segments': support_ref,
        'prediction_metrics_evaluated': prediction is not None,
        'scientific_accuracy_evidence': False, 'training_performed': False, 'source_and_score_files_unchanged': True,
        'elapsed_seconds': time.perf_counter() - started}
    write_json(output / 'receipt.json', receipt)
    print(__import__('json').dumps({'record_id': metadata['record_id'], 'counts': counts,
                                   'elapsed_seconds': receipt['elapsed_seconds']}, ensure_ascii=False), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=('fixed6', 'development'), default='fixed6')
    parser.add_argument('--development-snapshot', type=Path)
    parser.add_argument('--development-producer-root', type=Path)
    parser.add_argument('--fixed-selection', type=Path)
    parser.add_argument('--o1-snapshot', type=Path)
    parser.add_argument('--o1-producer-root', type=Path)
    parser.add_argument('--artifact-relocations', type=Path)
    parser.add_argument('--alignment-source', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--probe-record-id')
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    for name in ('development_snapshot', 'development_producer_root', 'artifact_relocations', 'alignment_source', 'output_root'):
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required')
    if args.preflight_only != bool(args.probe_record_id) or args.preflight_only and args.mode != 'development':
        parser.error('--preflight-only requires development mode and one explicit --probe-record-id')
    source_ref = {'path': str(args.development_snapshot.resolve()), 'sha256': sha(args.development_snapshot)}
    development = read_json(args.development_snapshot)
    entries = development_records(development)
    entry_index = {row['metadata']['record_id']: row for row in entries}
    if args.mode == 'fixed6':
        for name in ('fixed_selection', 'o1_snapshot', 'o1_producer_root'):
            if getattr(args, name) is None:
                parser.error(f'--{name.replace("_", "-")} is required for fixed6')
        selection = read_json(args.fixed_selection)
        if [row['record_id'] for row in selection['initial_examples']] != FIXED_IDS[:-1]:
            raise ValueError('The five historical fixed examples must not be reselected')
        records = selection['initial_examples'] + [{**selection['o1203'], 'record_id': 'O1'}]
        selection_ref = {'path': str(args.fixed_selection.resolve()), 'sha256': sha(args.fixed_selection)}
        expected_ids = FIXED_IDS
    else:
        if args.probe_record_id and args.probe_record_id not in entry_index:
            raise ValueError('The explicit probe record is absent from the development snapshot')
        expected_ids = list(entry_index)
        records = [{'record_id': rid} for rid in (expected_ids if not args.preflight_only else [args.probe_record_id])]
        selection_ref = source_ref
    args.output_root = args.output_root.resolve()
    tasks = args.output_root / 'tasks'
    tasks.mkdir(parents=True, exist_ok=True)
    relocation_ref = {'path': str(args.artifact_relocations.resolve()), 'sha256': sha(args.artifact_relocations)}
    generator_hash = sha(Path(__file__))
    alignment_ref = {'path': str(args.alignment_source.resolve()), 'sha256': sha(args.alignment_source)}
    candidate_sources = {'paper1_reference_attachment.py': sha(HERE / 'paper1_reference_attachment.py'),
                         'alignment_source': alignment_ref['sha256']}
    completed = []
    for record in records:
        rid = record['record_id']
        snapshot_path = (args.o1_snapshot if rid == 'O1' else args.development_snapshot).resolve()
        if rid == 'O1':
            snapshot = read_json(snapshot_path)
            entry = next(row for row in snapshot['records'] if row['kind'] == 'load' and row['metadata']['record_id'] == rid)
        else:
            entry = entry_index[rid]
        task = {'snapshot': {'path': str(snapshot_path), 'sha256': sha(snapshot_path)},
            'sample': {'path': str(snapshot_path.parent / entry['file']), 'sha256': entry['sha256']},
            'prediction': {'path': record['prediction'], 'sha256': record['prediction_sha256']} if args.mode == 'fixed6' else None,
            'metrics': {key: record[key] for key in ('rmse_w', 'mae_w')} if args.mode == 'fixed6' else None,
            'metadata': entry['metadata'], 'mode': args.mode, 'preflight_only': args.preflight_only,
            'output_root': str(args.output_root), 'generator_sha256': generator_hash, 'relocations': relocation_ref,
            'candidate_source_sha256': candidate_sources, 'alignment_source': alignment_ref,
            'producer_root': str((args.o1_producer_root if rid == 'O1' else args.development_producer_root).resolve())}
        task_path = tasks / f'{rid}.json'
        if task_path.exists() and read_json(task_path) != task:
            raise ValueError('Execution identity changed; use a new candidate output directory')
        if not task_path.exists():
            write_json(task_path, task)
        receipt_path = record_output(args.output_root, rid, preflight=args.preflight_only) / 'receipt.json'
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            for key in ('snapshot', 'sample', 'prediction', 'producer_root', 'generator_sha256', 'candidate_source_sha256', 'alignment_source', 'relocations', 'mode', 'preflight_only'):
                assert receipt[key] == task[key], key
            for key in ('attachment', 'support_segments'):
                if receipt[key]:
                    bound(receipt[key]['path'], receipt[key]['sha256'])
            bound(receipt['alignment_source']['path'], receipt['alignment_source']['sha256'])
        else:
            subprocess.run([sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--worker', str(task_path)],
                           check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        completed.append({'record_id': rid, 'receipt': {'path': str(receipt_path), 'sha256': sha(receipt_path)}})
    write_json(args.output_root / 'reference_manifest.json', {'schema': 'afc.reference_display_candidate_manifest.2',
        'status': 'single_record_preflight_only' if args.preflight_only else 'candidate_only',
        'mode': args.mode, 'preflight_only': args.preflight_only,
        'selection': selection_ref, 'selection_unchanged': True, 'development_snapshot': source_ref,
        'expected_record_ids': expected_ids, 'expected_records': len(expected_ids), 'completed_records': len(completed),
        'full_attachment_generation_complete': not args.preflight_only and len(completed) == len(expected_ids),
        'records': completed, 'generator_sha256': generator_hash,
        'candidate_source_sha256': candidate_sources, 'alignment_source': alignment_ref,
        'prediction_metrics_evaluated': args.mode == 'fixed6', 'O3_read': False, 'training_performed': False})

if __name__ == '__main__':
    main()
