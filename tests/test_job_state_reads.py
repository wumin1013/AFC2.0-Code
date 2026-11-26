"""Bounded Windows share-lock recovery, without suppressing persistent errors."""
from pathlib import Path
import json
import os
import sys
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from spindle_power.load_prediction.job_service import JobService, _read_text


def sharing_error(code):
    error = PermissionError(13, 'synthetic Windows read sharing conflict')
    error.winerror = code
    return error


@pytest.mark.skipif(os.name != 'nt', reason='Windows-only sharing retry contract')
@pytest.mark.parametrize('code', [5, 32, 33])
def test_transient_windows_read_conflict_retries_then_succeeds(code):
    with patch.object(Path, 'read_text', side_effect=[sharing_error(code), '{"ok":true}']) as read:
        with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
            assert json.loads(_read_text(Path('unused'))) == {'ok': True}
    assert read.call_count == 2
    sleep.assert_called_once_with(.01)


@pytest.mark.skipif(os.name != 'nt', reason='Windows-only sharing retry contract')
def test_persistent_windows_permission_failure_raises_original_error():
    error = sharing_error(32)
    with patch.object(Path, 'read_text', side_effect=error) as read:
        with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
            with pytest.raises(PermissionError) as failure:
                _read_text(Path('unused'))
    assert failure.value is error and read.call_count == 7
    assert [entry.args[0] for entry in sleep.call_args_list] == [.01, .02, .04, .08, .1, .1]


def test_other_permission_error_is_not_retried():
    with patch.object(Path, 'read_text', side_effect=sharing_error(123)) as read:
        with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
            with pytest.raises(PermissionError):
                _read_text(Path('unused'))
    assert read.call_count == 1
    sleep.assert_not_called()


def test_missing_file_is_not_retried(tmp_path):
    with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
        with pytest.raises(FileNotFoundError):
            _read_text(tmp_path / 'missing.json')
    sleep.assert_not_called()


def service_without_worker(directory):
    service = JobService.__new__(JobService)
    service.directory = directory
    return service


def test_corrupt_state_json_is_not_swallowed(tmp_path):
    directory = tmp_path / 'job1'
    directory.mkdir()
    (directory / 'state.json').write_text('{broken', encoding='utf-8')
    with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
        with pytest.raises(json.JSONDecodeError):
            service_without_worker(tmp_path).get('job1')
    sleep.assert_not_called()


@pytest.mark.skipif(os.name != 'nt', reason='Windows-only sharing retry contract')
def test_get_retries_state_and_events_read_conflicts(tmp_path):
    directory = tmp_path / 'job1'
    directory.mkdir()
    (directory / 'state.json').write_text('{"job_id":"job1","status":"running"}', encoding='utf-8')
    (directory / 'events.jsonl').write_text('{"epoch":1}\n', encoding='utf-8')
    original, calls = Path.read_text, {}
    def conflict_once(path, *args, **kwargs):
        calls[path.name] = calls.get(path.name, 0) + 1
        if calls[path.name] == 1:
            raise sharing_error(32)
        return original(path, *args, **kwargs)
    with patch.object(Path, 'read_text', conflict_once):
        with patch('spindle_power.load_prediction.job_service.time.sleep'):
            state = service_without_worker(tmp_path).get('job1')
    assert state['status'] == 'running' and state['events'] == [{'epoch': 1}]
    assert calls == {'state.json': 2, 'events.jsonl': 2}


def test_corrupt_events_json_is_not_swallowed(tmp_path):
    directory = tmp_path / 'job1'
    directory.mkdir()
    (directory / 'state.json').write_text('{"job_id":"job1","status":"running"}', encoding='utf-8')
    (directory / 'events.jsonl').write_text('{broken\n', encoding='utf-8')
    with patch('spindle_power.load_prediction.job_service.time.sleep') as sleep:
        with pytest.raises(json.JSONDecodeError):
            service_without_worker(tmp_path).get('job1')
    sleep.assert_not_called()
