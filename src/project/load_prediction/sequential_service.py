"""Score, seal, release, then retrain: a separate service for later events.

Initial development remains governed by TrainingService's original 450-record
contract. This service never schedules jobs and never adds data to an old run.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import uuid

import pandas as pd

from .common import file_hash, save_json
from .samples import SampleSet
from .training_checkpoint import digest, sample_fingerprint, source_fingerprints
from .training_service import NOMINAL_ROUTE, evaluate_records, compare_evaluation_artifacts, train_full


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def bound_file(reference):
    path = Path(reference['path']).resolve()
    if file_hash(path) != reference['sha256']:
        raise ValueError(f'文件身份不一致：{path}')
    return path


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def load_snapshot(ref, *, group_id=None):
    path = bound_file(ref)
    manifest = read_json(path)
    if manifest.get('synthetic') is not False or digest({k:v for k,v in manifest.items() if k != 'snapshot_id'}) != manifest['snapshot_id']:
        raise ValueError('序贯正式实验必须使用身份完整的非合成快照')
    rows = [row for row in manifest['records'] if row['kind'] == 'load']
    if not rows:
        raise ValueError('快照缺少负载记录')
    samples = []
    for row in rows:
        file = bound_file(dict(path=str(path.parent / row['file']), sha256=row['sha256']))
        frame = pd.read_csv(file, float_precision='round_trip')
        for column, dtype in row['dtypes'].items():
            if dtype == 'bool':
                frame[column] = frame[column].astype(bool)
        meta = dict(row['metadata'])
        if group_id is not None:
            if meta.get('group_id') != group_id or meta.get('round_id') is not None or meta.get('dataset_role') != 'sequential_event':
                raise ValueError('后续组身份不符或冒用开发轮次')
            if any(meta.get(key) is not True for key in ('complete', 'alignment_reliable', 'tool_healthy', 'no_alarm')):
                raise ValueError('后续组采集质量尚未全部核验')
            if not frame.valid_main.any() or not frame.valid_residual.any():
                raise ValueError('后续组缺少两阶段有效监督')
        samples.append(SampleSet(frame, meta))
    if group_id is not None:
        from .record_idle import verify_record_idle_evidence, verify_prepared_record_idle
        qualification = manifest['source_snapshot']['base_validation']
        check = verify_record_idle_evidence(qualification, [s.metadata for s in samples], base_dir=path.parent)
        if not check['valid']:
            raise ValueError('；'.join(check['issues']))
        # Resolve frozen relative evidence before rechecking the prepared mask.
        qualification = {**qualification, 'report_path': check['files']['report']['path']}
        for sample in samples:
            verify_prepared_record_idle(sample, qualification)
    return samples, manifest


class SequentialService:
    def __init__(self, case_path, output_dir):
        self.case_path = Path(case_path).resolve()
        self.case = read_json(self.case_path)
        self.output = Path(output_dir).resolve()
        self.seal_path = bound_file(self.case['k0_seal'])
        self.seal = read_json(self.seal_path)
        if self.seal.get('status') != 'accepted' or self.seal.get('schema') != 'paper1.k0_seal.1':
            raise ValueError('K0未正式验收封存')
        self.order = [g['group_id'] for g in self.case['groups']]
        if self.order != ['O1', 'O2', 'O3']:
            raise ValueError('当前案例必须显式登记O1→O2→O3')
        self.groups = {g['group_id']: g for g in self.case['groups']}
        self.k0_path = self.seal_path.parent / 'load.pt'
        self.scoring = read_json(self.seal_path.parent / 'scoring_parameters.json')['parameters']
        if self.scoring != self.seal['frozen_scoring']:
            raise ValueError('评分参数与K0封存不一致')
        if self.case['training_rule'] != dict(capacity=self.seal['selected'], epochs=self.seal['full_epochs'],
                initialization='fresh', selection=False, early_stopping=False, device='cpu', cpu_threads=4, cpu_interop_threads=1):
            raise ValueError('累计训练必须沿用K0容量及固定epoch规则')
        self.check_k0()

    def check_k0(self):
        if file_hash(self.k0_path) != self.seal['k0']['model_files']['load.pt']:
            raise ValueError('K0权重已改变')
        bound_file(self.case['k0_seal'])

    @contextmanager
    def exclusive(self):
        self.output.mkdir(parents=True, exist_ok=True)
        lock = self.output / 'operation.lock'
        with lock.open('x', encoding='utf-8') as stream:
            json.dump(dict(pid=os.getpid(), case=reference(self.case_path)), stream)
        try:
            case_identity = self.output / 'case_identity.json'
            current = reference(self.case_path)
            if case_identity.exists() and read_json(case_identity) != current:
                raise ValueError('不同案例身份不能复用输出目录')
            if not case_identity.exists():
                save_json(case_identity, current)
            self.check_k0()
            yield
        finally:
            lock.unlink()

    def _receipt(self, directory):
        path = Path(directory) / 'receipt.json'
        value = read_json(path)
        if value.get('status') != 'sealed':
            raise ValueError('产物尚未封存')
        for item in value['artifacts'].values():
            bound_file(item)
        if value['case'] != reference(self.case_path):
            raise ValueError('产物案例身份不一致')
        return value

    def _seal(self, directory, value):
        directory = Path(directory)
        artifacts = {p.relative_to(directory).as_posix(): reference(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'receipt.json'}
        receipt = dict(status='sealed', case=reference(self.case_path), sealed_at=datetime.now(timezone.utc).isoformat(),
                       artifacts=artifacts, **value)
        save_json(directory / 'receipt.json', receipt)
        return receipt

    def _model(self, name):
        from .models import load_models
        if name == 'K0':
            self.check_k0()
            path = self.k0_path
        else:
            self._receipt(self.output / 'models' / name)
            path = self.output / 'models' / name / 'weights' / 'load.pt'
        model = load_models(path, device='cpu')
        model.assert_ready()
        return model, reference(path)

    def _released(self, group_id):
        path = self.output / 'released' / f'{group_id}.json'
        value = read_json(path)
        score = self.output / 'scores' / group_id
        self._receipt(score)
        if value != dict(group_id=group_id, score=reference(score / 'receipt.json'), snapshot=self.groups[group_id]['snapshot']):
            raise ValueError('释放身份与已评分组不一致')
        return reference(path)

    def score(self, group_id):
        if group_id not in self.groups:
            raise ValueError('未登记组')
        with self.exclusive():
            index = self.order.index(group_id)
            for previous in self.order[:index]:
                self._released(previous)
            names = ['K0'] if index == 0 else ['K0', f'K{index}']
            models = {name: self._model(name) for name in names}
            destination = self.output / 'scores' / group_id
            if (destination / 'receipt.json').exists():
                return self._receipt(destination)
            samples, manifest = load_snapshot(self.groups[group_id]['snapshot'], group_id=group_id)
            # A fresh attempt preserves all failed predictions and partial files.
            attempt = self.output / 'attempts' / f'{group_id}_{uuid.uuid4().hex}'
            attempt.mkdir(parents=True)
            try:
                paths = {}
                for name, (model, _) in models.items():
                    paths[name] = attempt / name
                    evaluate_records(model, samples, feed_mode='programmed', memory=None,
                        metric_parameters=self.scoring, output_dir=paths[name])
                if len(paths) > 1:
                    compare_evaluation_artifacts(paths, output_dir=attempt / 'common')
                # Move before sealing so every artifact reference is permanent.
                destination.parent.mkdir(parents=True, exist_ok=True)
                attempt.rename(destination)
                if len(paths) > 1:
                    comparison = read_json(destination / 'common/comparison.json')
                    comparison['source_directories'] = {n:str(destination/n) for n in names}
                    save_json(destination / 'common/comparison.json', comparison)
                self.check_k0()
                return self._seal(destination, dict(group_id=group_id, snapshot=self.groups[group_id]['snapshot'],
                    snapshot_id=manifest['snapshot_id'], models={n:ref for n,(_,ref) in models.items()},
                    scoring_parameters=self.scoring, memory=False, code=source_fingerprints(),
                    terminal_case=index == 2, release_permitted=index < 2))
            except Exception as exc:
                failure_dir = attempt if attempt.exists() else destination
                save_json(failure_dir / 'failure.json', dict(status='failed', group_id=group_id,
                    error=f'{type(exc).__name__}: {exc}', release_permitted=False))
                raise

    def release(self, group_id):
        if group_id not in self.order[:2]:
            raise ValueError('O3评分后结束；不释放新的未验证更新')
        with self.exclusive():
            for previous in self.order[:self.order.index(group_id)]:
                self._released(previous)
            receipt = self._receipt(self.output / 'scores' / group_id)
            if receipt['snapshot'] != self.groups[group_id]['snapshot'] or receipt['release_permitted'] is not True:
                raise ValueError('未评分或不同数据身份不能释放')
            value = dict(group_id=group_id, score=reference(self.output / 'scores' / group_id / 'receipt.json'),
                         snapshot=self.groups[group_id]['snapshot'])
            path = self.output / 'released' / f'{group_id}.json'
            if path.exists():
                if read_json(path) != value:
                    raise ValueError('重复释放身份冲突')
            else:
                save_json(path, value)
            return value

    def train(self, released_groups, *, resume=False, progress=None):
        groups = list(released_groups)
        if groups not in [self.order[:1], self.order[:2]]:
            raise ValueError('训练仅接受已评分的连续前缀O1或O1+O2')
        with self.exclusive():
            releases = [self._released(group) for group in groups]
            name = f'K{len(groups)}'
            directory = self.output / 'models' / name
            if (directory / 'receipt.json').exists():
                return self._receipt(directory)
            initial, manifest = load_snapshot(self.case['initial_snapshot'])
            if manifest['snapshot_id'] != self.seal['snapshot_id'] or len(initial) != 450:
                raise ValueError('累计训练初始D0与封存K0不一致')
            # Keep the existing exact 90-round/five-category check unchanged.
            from .samples import grouped_folds
            grouped_folds([s.metadata for s in initial], require_complete=True)
            samples = list(initial)
            for group in groups:
                later, _ = load_snapshot(self.groups[group]['snapshot'], group_id=group)
                samples.extend(later)
            if len({s.metadata['record_id'] for s in samples}) != len(samples):
                raise ValueError('累计样本记录身份重复')
            k0, _ = self._model('K0')
            training = {**k0.metadata['config'], 'device': 'cpu', 'cpu_threads': 4, 'cpu_interop_threads': 1}
            candidate = dict(id=self.seal['selected'], load_model=asdict(k0.config), load_training=training)
            development = dict(route=NOMINAL_ROUTE, selected=candidate, full_epochs=self.seal['full_epochs'])
            identity = dict(case=reference(self.case_path), releases=releases, data=sample_fingerprint(samples),
                            development=development, code=source_fingerprints())
            run = self.output / 'runs' / name
            identity_path = run / 'identity.json'
            if identity_path.exists():
                if not resume or read_json(identity_path) != identity:
                    raise ValueError('恢复要求同数据、源码、配置身份；不能追加数据恢复旧任务')
            else:
                if resume:
                    raise ValueError('没有对应的序贯运行可恢复')
                save_json(identity_path, identity)
            save_json(run / 'state.json', dict(status='running', model=name, released_groups=groups))
            def report(event):
                with (run / 'events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + '\n')
                if progress:
                    progress(event)
            try:
                _, bundle = train_full(samples, [], development, output_dir=directory / 'weights',
                    checkpoint_dir=run / 'checkpoints', resume=resume, progress=report)
                for stage, budget in self.seal['full_epochs'].items():
                    history = bundle.metadata['stages'][stage]
                    if history['early_stopping'] or [row['epoch'] for row in history['history']] != list(range(1,budget+1)):
                        raise ValueError('累计训练固定完整epoch验收失败')
                if identity['code'] != source_fingerprints():
                    raise ValueError('训练过程中源码发生变化')
                self.check_k0()
                model_id = digest(dict(identity=identity, weights=file_hash(directory/'weights/load.pt')))
                result = self._seal(directory, dict(model_id=model_id, version=name, released_groups=groups,
                    data_identity=identity['data'], code=identity['code'], training_rule=self.case['training_rule'],
                    normalization=bundle.normalization, scoring_parameters=self.scoring,
                    old_memory_reused=False, memory_rebuild_required=True))
                save_json(run / 'state.json', dict(status='succeeded', model_id=model_id, receipt=reference(directory/'receipt.json')))
                return result
            except Exception as exc:
                save_json(run / 'state.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', released_groups=groups))
                raise

    def memory_sensitivity(self, group_id):
        """Rebuild each model's memory from released O history; never update on target."""
        from .memory import PositionMemory
        if group_id not in self.groups:
            raise ValueError('未登记组')
        with self.exclusive():
            self._receipt(self.output / 'scores' / group_id)
            index = self.order.index(group_id)
            history = self.order[:index]
            releases = [self._released(group) for group in history]
            destination = self.output / 'memory_sensitivity' / group_id
            if (destination / 'receipt.json').exists():
                return self._receipt(destination)
            destination.mkdir(parents=True, exist_ok=False)
            def declared(samples, order):
                # Copies carry analysis assumptions; frozen facts stay unknown.
                result = deepcopy(samples)
                for sample in result:
                    sample.metadata.update(tool_session='declared:O1203_same_tool_pool',actual_event_order=order,
                        memory_analysis_mode='declared_replay_sensitivity_only')
                return result
            names = ['K0'] if index == 0 else ['K0', f'K{index}']
            for name in names:
                model, _ = self._model(name)
                memory = PositionMemory(minimum_repeats=2)
                for order, group in enumerate(history):
                    samples, _ = load_snapshot(self.groups[group]['snapshot'],group_id=group)
                    evaluate_records(model,declared(samples,order),memory=memory,metric_parameters=self.scoring,
                        output_dir=destination/name/'history'/group)
                memory.save(destination/name/'memory_before_target.json')
                before = digest(asdict(memory))
                target, _ = load_snapshot(self.groups[group_id]['snapshot'],group_id=group_id)
                evaluate_records(model,declared(target,index),memory=memory,metric_parameters=self.scoring,
                    update_memory=False,output_dir=destination/name/'target')
                if digest(asdict(memory)) != before:
                    raise ValueError('目标评分错误更新了重建记忆')
                compare_evaluation_artifacts(dict(without_memory=self.output/'scores'/group_id/name,
                    with_memory=destination/name/'target'),output_dir=destination/name/'common')
            self.check_k0()
            return self._seal(destination,dict(group_id=group_id,releases=releases,models=names,
                interpretation='declared_tool_pool_and_replay_order_sensitivity_only',
                old_memory_reused=False,rebuilt_from=history,minimum_repeats=2,target_memory_updated=False,
                initial_program_memory='No matching O1203 program positions in D0',code=source_fingerprints()))
