"""Full-position parser and assembly regressions without external data."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import tempfile, unittest, types
import numpy as np
import pandas as pd
import torch
from spindle_power.spindle_runtime import model as _module
from spindle_power.spindle_runtime.full_position import (prepare_source_timeline, parse_research_nc,
    combine_command_power, combine_source_power, DIRECT_SOURCE_POSITION_POLICY, OUT_COLS,
    predict_prepared_positions, prepare_full_positions)
from spindle_power.spindle_runtime.startup_events import attach_startup_curves, isolated_startup_inputs

class TimelineTests(unittest.TestCase):
    def test_g04_x_is_dwell_not_position(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'dwell.nc'
            path.write_text('G0 X0 Y0 Z0\nG1 X10 F600\nG4 X2\nG1 X20\n', encoding='utf-8')
            _, segments, _, _, _ = parse_research_nc(path)
        self.assertEqual([segment.line for segment in segments], [2, 4])
        self.assertEqual([segment.length for segment in segments], [10., 10.])

    def test_unknown_unlisted_motion_duration_is_recorded(self):
        source = pd.DataFrame(dict(line=[1], path_mm=[0.], source_feed_mm_min=[600.]))
        _, meta = prepare_source_timeline(['G0 X0', 'G28 Z0'], [], source, {})
        self.assertTrue(any(row.get('line') == 2 and row['kind'] == 'unestimated_motion_duration'
                            for row in meta['assumptions']))

    def test_braking_command_zero_does_not_remove_unknown_cutting(self):
        result = pd.DataFrame(dict(ap_mm=[1.,1.], ae_mm=[1.,1.], source_spindle_rpm=[0.,0.],
            missing_reason=['unknown_geometry']*2, mapping_method=['unavailable']*2))
        for col in OUT_COLS:
            result[col] = np.nan
        prepared = dict(timing=dict(mode='test', command_at_source_rpm=[0.,0.],
                stopped_before_first_start=[True,False]), geometry_audit=[],
            timeline=dict(time_s=np.array([[0.,1.,2.]]),commands_rpm=np.array([[0.,12000.,0.]]),
                initial_speed_rpm=np.array([0.]),point_time_s=np.array([0.,2.])))
        actual, _ = combine_source_power(prepared,result,np.array([50.,100.]),{},{},{},
                                          curve_max=10000.,domain_min=3000.,extrapolate=True)
        self.assertEqual(actual.loc[0,'total_power_w'],50.)
        self.assertEqual(actual.loc[1,'prediction_status'],'blocked_cutting_geometry')
        self.assertTrue(np.isnan(actual.loc[1,'cutting_power_w']))

    def test_supplied_requires_all_source_points_and_order(self):
        source = pd.DataFrame({'line': [1, 2]})
        context = dict(prediction_timeline=dict(point_mapping='source_process_order',
            time_s=[0., 1., 2.], commands_rpm=[0., 12000., 12000.], initial_speed_rpm=0.,
            point_time_s=[0., 2.]))
        tensors, meta = prepare_source_timeline([], [], source, context)
        np.testing.assert_array_equal(tensors['point_time_s'], [0., 2.])
        self.assertEqual(meta['mode'], 'supplied_source_timeline')
        context['prediction_timeline']['point_time_s'] = [1.]
        with self.assertRaises(ValueError):
            prepare_source_timeline([], [], source, context)

    def test_reverse_command_rejected(self):
        source = pd.DataFrame(dict(line=[1], path_mm=[0.], source_feed_mm_min=[1000.]))
        with self.assertRaisesRegex(ValueError, 'M4'):
            prepare_source_timeline(['S5000 M4'], [], source, {})

    def test_unknown_motion_timing_is_explicit(self):
        source = pd.DataFrame(dict(line=[2], path_mm=[10.], source_feed_mm_min=[600.]))
        tensors, meta = prepare_source_timeline(['S12000 M3', 'G0 X10'], [], source, {})
        self.assertAlmostEqual(tensors['point_time_s'][0], 1.)
        self.assertTrue(any(a['kind']=='rapid_feed_mm_min' for a in meta['assumptions']))
        self.assertFalse(meta['timing_is_measured'])
        self.assertEqual(meta['command_at_source_rpm'], [12000.])

    def test_spindle_wait_precedes_motion_and_m5_overrides_stale_source_speed(self):
        source = pd.DataFrame(dict(line=[1, 2], path_mm=[10., 10.], source_feed_mm_min=[600.,600.]))
        tensors, meta = prepare_source_timeline(['G0 X10 S12000 M3', 'M5'], [], source,
            {'prediction_policy': {'spindle_wait_s': 2., 'rapid_feed_mm_min': 600.}})
        np.testing.assert_array_equal(tensors['point_time_s'], [3., 3.])
        self.assertEqual(meta['command_at_source_rpm'], [12000., 0.])
        self.assertEqual(meta['events'][-1]['time_s'], 3.)

    def test_supplied_command_sampling_is_zero_order_hold(self):
        source = pd.DataFrame({'line': [1]})
        tensors, meta = prepare_source_timeline([], [], source, {'prediction_timeline':
            dict(point_mapping='source_process_order', time_s=[0.,1.,2.],
                 commands_rpm=[0.,12000.,12000.], initial_speed_rpm=0., point_time_s=[.5])})
        self.assertEqual(meta['command_at_source_rpm'], [0.])

    def test_command_domain_uses_nc_line_and_point_order_with_no_timeline(self):
        program = ['G0 X0 Y0 Z0', 'S10000 M3', 'IF [#1380 GT 0] PRINT [OFFSET]', 'S12000',
                   'M5', 'S9000', 'M3', 'M4', 'M30']
        point_lines = [1, 2, 2, 3, 4, 5, 6, 7, 8, 9]
        with tempfile.TemporaryDirectory() as folder:
            nc = Path(folder)/'program.nc'
            process = Path(folder)/'process.txt'
            nc.write_text('\n'.join(program), encoding='utf-8')
            rows = [f'{i} {line} 0 0 600 0 {i} 9999 {program[line - 1]}'
                    for i, line in enumerate(point_lines)]
            process.write_text('header\n' + '\n'.join(rows), encoding='utf-8')
            prepared = prepare_full_positions(nc, process, {'prediction_timeline': {'invalid': True}},
                position_policy=DIRECT_SOURCE_POSITION_POLICY, command_domain=True)
        self.assertIsNone(prepared['timeline'])
        self.assertEqual(prepared['timing']['command_at_source_rpm'],
                         [0., 10000., 10000., 10000., 12000., 0., 0., 9000., None, 0.])
        self.assertEqual(prepared['source'].line.tolist(), point_lines)
        self.assertEqual(prepared['source'].point_index0.tolist(), [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        starts = prepared['spindle_start_commands']
        self.assertEqual([(event['line'], event['target_rpm']) for event in starts], [(2, 10000.), (7, 9000.)])
        self.assertEqual(starts[0]['display_before_source'],
                         {'source_index': 1, 'line': 2, 'source_line_occurrence': 0, 'point_index0': 0})
        self.assertIsNone(starts[0]['first_engaged_source'])

    def test_startup_curves_sample_only_the_calibrated_event(self):
        prepared = {'spindle_start_commands': [dict(line=2, target_rpm=10000.), dict(line=9, target_rpm=12000.),
                                               dict(line=18, target_rpm=10000.)]}
        profile = {'base_event_calibration': {'startup': [3000., 10000.]}}
        phases = np.array([[0., 1., 2., 3., 4.], [0., 2., 3., 4., 5.]])
        evaluations = []

        def evaluate(target, elapsed):
            inputs = isolated_startup_inputs(target, elapsed)
            np.testing.assert_array_equal(inputs['commands_rpm'], np.full((65, 3), 10000.))
            np.testing.assert_array_equal(inputs['initial_speed_rpm'], np.zeros(65))
            np.testing.assert_allclose(np.diff(inputs['time_s'], axis=1)[:, 0],
                                       np.diff(inputs['time_s'], axis=1)[:, 1])
            np.testing.assert_array_equal(inputs['point_time_s'][:, 0], elapsed)
            evaluations.append(elapsed)
            return 250. + 10.*elapsed

        metadata = {}
        attach_startup_curves(prepared, metadata, profile, np.array([3000., 10000.]), phases, evaluate)
        events = metadata['spindle_startup_events']
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(events[0]['curve']['profile_duration_s'], 5.)
        self.assertEqual(events[0]['curve']['peak_w'], 300.)
        self.assertEqual(events[0]['curve']['cutting_power_w'], 0.)
        self.assertEqual(events[1]['status'], 'outside_startup_calibration')
        self.assertEqual(events[2]['curve'], events[0]['curve'])

    def test_command_power_preserves_cutting_and_flags_unavailable_or_extrapolated_base(self):
        source = pd.DataFrame(dict(ap_mm=[0., 1., 1., 0., 0.], ae_mm=[0., 1., 1., 0., 0.],
            missing_reason=['', '', 'unknown_geometry', '', ''], mapping_method=[''] * 5))
        source[OUT_COLS] = [[np.nan] * 5, [330., 320., 340., 250., 80.],
                            [np.nan] * 5, [np.nan] * 5, [np.nan] * 5]
        prepared = dict(timeline=None, geometry_audit=[], position_policy=DIRECT_SOURCE_POSITION_POLICY,
            timing=dict(mode='command_domain', command_at_source_rpm=[0., 10000., 10000., 12000., None]))
        profile = dict(base_curve=dict(speeds_rpm=[0., 8000., 10000.], powers_w=[0., 200., 250.]))
        for extrapolate, expected in ((False, np.nan), (True, 300.)):
            with self.subTest(extrapolate=extrapolate):
                actual, meta = combine_command_power(prepared, source.copy(), {}, {}, profile, extrapolate=extrapolate)
                np.testing.assert_allclose(actual.base_power_w, [0., 250., 250., expected, np.nan])
                self.assertEqual(actual.loc[1, 'cutting_power_w'], 80.)
                self.assertEqual(actual.loc[1, 'total_power_w'], 330.)
                self.assertTrue(np.isnan(actual.loc[2, 'total_power_w']))
                self.assertEqual(actual.loc[4, 'missing_reason'], 'command_base_unavailable')
                self.assertEqual(actual.loc[3, 'base_is_extrapolation'], extrapolate)
                self.assertTrue(actual.point_time_s.isna().all())
                self.assertFalse(meta['base_execution']['timeline_used'])

    def test_negative_tail_rejected(self):
        source = pd.DataFrame(dict(line=[1], path_mm=[0.], source_feed_mm_min=[600.]))
        with self.assertRaisesRegex(ValueError, 'tail_duration'):
            prepare_source_timeline(['S12000 M3'], [], source, {'prediction_policy': {'tail_duration_s': -1}})

    def test_unavailable_dynamic_is_not_reported_as_success(self):
        class InvalidBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.base_speeds_rpm = torch.tensor([0., 10000.])
                self.dynamic_base = types.SimpleNamespace(
                    sample_points=_module.DynamicBasePower.sample_points,
                    startup_domain_rpm=torch.tensor([3000., 10000.]))
            def predict_dynamic_base(self, time, commands, initial, *, extrapolate):
                return torch.full_like(time, float('nan')), commands, torch.zeros_like(time, dtype=torch.bool)
        prepared = dict(source=pd.DataFrame(dict(line=[1], ap_mm=[0.], ae_mm=[0.],
            source_spindle_rpm=[0.], missing_reason=['unknown_start'], segment_mm=[np.nan])),
            grid=pd.DataFrame(), segments=[], reasons={1:'unknown_start'},
            timeline=dict(time_s=np.array([[0.,1.,2.]]), commands_rpm=np.array([[0.,0.,0.]]),
                initial_speed_rpm=np.array([0.]), point_time_s=np.array([0.])),
            timing=dict(mode='supplied_source_timeline',command_at_source_rpm=[0.]), geometry_audit=[])
        result, meta = predict_prepared_positions(prepared, InvalidBase(), {})
        self.assertFalse(meta['complete'])
        self.assertEqual(result.loc[0,'prediction_status'], 'blocked_dynamic_base')

