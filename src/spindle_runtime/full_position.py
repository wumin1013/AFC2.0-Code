"""Full-position research prediction with explicit command-time assumptions.

The source files remain read-only. Cutting geometry checks are preserved;
base power is computed independently for every process position.
"""
from __future__ import annotations
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch

try:
    from afc_vendor.load_prediction.feed import Segment, _arc
    from afc_vendor.load_prediction.feed_samples import _tangent, _curvature, _angle
    from afc_vendor.load_prediction.process_coordinates import reconstruct_process_coordinates, source_distance_compatible, source_cell_indices
    from afc_vendor.load_prediction.window_plan import prospective_residual_context, window_batches
    from afc_vendor.load_prediction.samples import PHYSICS_COLUMNS, NOMINAL_GEOMETRY_COLUMNS
    from afc_vendor.load_prediction.measurement import detect_encoding
except ModuleNotFoundError as exc:
    if exc.name not in {'afc_vendor', 'afc_vendor.load_prediction'}:
        raise
    from load_prediction.feed import Segment, _arc
    from load_prediction.feed_samples import _tangent, _curvature, _angle
    from load_prediction.process_coordinates import reconstruct_process_coordinates, source_distance_compatible, source_cell_indices
    from load_prediction.neural import prospective_residual_context, window_batches
    from load_prediction.samples import PHYSICS_COLUMNS, NOMINAL_GEOMETRY_COLUMNS
    from load_prediction.measurement import detect_encoding

STATUS = 'research_prediction_with_command_time_estimate'
IN_COLS = PHYSICS_COLUMNS + ['curvature_per_mm', 'block_length_mm'] + NOMINAL_GEOMETRY_COLUMNS
OUT_COLS = ['total_power_w','total_lower_w','total_upper_w','base_power_w','cutting_power_w']
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
WORD = rf'([A-Z])\s*({NUMBER})'

def parse_research_nc(path):
    path = Path(path)
    raw_lines = path.read_text(encoding=detect_encoding(path)).splitlines()
    xyz = np.full(3, np.nan)
    rotation = dict(A=np.nan, B=np.nan, C=np.nan)
    absolute, motion, feed, spindle_setting, spindle, group = True, 0, np.nan, 0., 0., 0
    requires_tcp = any(re.search(r'G\s*(?:43\.4|68\.2)(?!\d)', line.upper()) for line in raw_lines)
    tcp, tilted = not requires_tcp, False
    variables, segments, info, reasons, events = {}, [], {}, {}, []
    accepted_g = {0,1,2,3,4,5.1,17,21,28,40,43.4,49,53,53.1,54,55,56,57,58,59,68.2,69,90,91,94,94.6}
    accepted_m = {0,1,3,4,5,6,8,9,30,41,46,99,141,146}
    for line_no, raw in enumerate(raw_lines, 1):
        line = re.sub(r'\([^)]*\)', '', raw).split(';',1)[0].strip().upper()
        if not line or line=='%' or re.fullmatch(r'%[0-9]+|[ON]\d+',line):
            continue
        assignment = re.fullmatch(rf'#(\d+)\s*=\s*({NUMBER})', line)
        if assignment:
            variables[assignment[1]]=float(assignment[2]); continue
        try:
            def expand(match):
                if match[1] not in variables:
                    raise ValueError('unknown_macro_variable')
                return str(variables[match[1]])
            line=re.sub(r'#(\d+)',expand,line)
            pairs=re.findall(WORD,line)
            if re.sub(WORD,'',line).strip():
                raise ValueError('uninterpreted_nc_syntax')
            codes={float(v) for k,v in pairs if k=='G'}
            mcodes={int(float(v)) for k,v in pairs if k=='M'}
            words={k:float(v) for k,v in pairs if k not in ('G','M','N','O')}
            if codes-accepted_g or mcodes-accepted_m:
                raise ValueError('uninterpreted_nc_directive')
            if 90 in codes: absolute=True
            if 91 in codes: absolute=False
            for g in (0,1,2,3):
                if g in codes: motion=g
            if 'F' in words: feed=words['F']
            if 'S' in words:
                spindle_setting=words['S']
                if spindle>0: spindle=spindle_setting
            if 3 in mcodes: spindle=spindle_setting
            if 5 in mcodes: spindle=0.
            if mcodes & {0,1,3,5,6,41,46,141,146}: group+=1
            if 6 in mcodes: xyz[:]=np.nan
            for k in 'ABC':
                if k in words: rotation[k]=words[k] if absolute else rotation[k]+words[k]
            reset = bool(codes & {28,49,53,53.1,68.2,69})
            if 68.2 in codes: tilted=True
            if 69 in codes: tilted=False
            if 49 in codes: tcp=not requires_tcp
            if 43.4 in codes:
                tcp=True; group+=1; xyz[:]=np.nan
            if reset:
                group+=1; xyz[:]=np.nan
                reasons[line_no]='coordinate_reset_or_machine_positioning'
                events.append(dict(line=line_no,codes=sorted(codes),local_tilted_frame=tilted))
                # XYZ on G68.2 is a frame origin, not a tool-tip move.
                continue
            if 4 in codes:
                # G04 X is dwell duration, never an X-axis target.
                reasons[line_no]='dwell_without_geometry'
                continue
            has_xyz=any(k in words for k in 'XYZ')
            if has_xyz:
                target=xyz.copy()
                for i,k in enumerate('XYZ'):
                    if k in words: target[i]=words[k] if absolute else target[i]+words[k]
                if not np.isfinite(xyz).all() or not np.isfinite(target).all():
                    reasons[line_no]='unknown_start_endpoint_anchor_only'
                else:
                    if motion in (2,3): center,sweep=_arc(xyz,target,words,motion==2)
                    else: center,sweep=None,0.
                    seg=Segment(line_no,xyz.copy(),target.copy(),motion,float(feed),float(spindle),group,center,sweep)
                    if seg.length<=1e-9:
                        reasons[line_no]='zero_xyz_motion'
                    elif motion==0:
                        reasons[line_no]='rapid_motion_not_scored'
                    elif not tcp:
                        reasons[line_no]='tcp_mode_not_active'
                    elif not np.isfinite(feed) or feed<=0:
                        reasons[line_no]='unknown_or_invalid_program_feed'
                    else:
                        segments.append(seg)
                        info[line_no]=dict(a_deg=rotation['A'],b_deg=rotation['B'],c_deg=rotation['C'],local_tilted_frame=tilted)
                xyz=target
                if motion==0: group+=1
            elif any(k in words for k in 'ABC'):
                group+=1
                reasons[line_no]='rotary_only_no_xyz_path'
            if mcodes & {30,99}: break
        except (ValueError,KeyError) as exc:
            group+=1; xyz[:]=np.nan
            reasons[line_no]=str(exc)
            events.append(dict(line=line_no,error=str(exc)))
    return raw_lines,segments,info,reasons,events

def read_process(path, raw_nc):
    path = Path(path)
    text=path.read_text(encoding=detect_encoding(path))
    rows=[]
    previous=None; point=0; occurrence=Counter()
    for file_line,line in enumerate(text.splitlines()[1:],2):
        if not line.strip(): continue
        fields=line.split(maxsplit=8)
        vals=[float(v) for v in fields[:8]]
        if not np.isfinite(vals).all(): raise ValueError('nonfinite_process_source')
        idx,nc_line,ap,ae,feed,mrr,distance,rpm=vals
        nc_line=int(nc_line)
        if re.sub(r'\s+',' ', fields[8].strip()) != re.sub(r'\s+',' ', raw_nc[nc_line-1].strip()):
            raise ValueError('source_process_and_physical_nc_line_mismatch')
        if nc_line!=previous:
            point=0; occurrence[nc_line]+=1
        rows.append(dict(source_index=int(idx),line=nc_line,point_index0=point,
            line_point_label=f'{nc_line}.{point}',source_file_line=file_line,
            path_mm=distance,source_s_token=fields[6],ap_mm=ap,ae_mm=ae,
            source_feed_mm_min=feed,source_mrr_mm3_min=mrr,source_spindle_rpm=rpm,
            nc_text=fields[8],source_line_occurrence=occurrence[nc_line]-1))
        point+=1; previous=nc_line
    result=pd.DataFrame(rows)
    if (np.diff(result.path_mm.to_numpy())<0).any(): raise ValueError('source_s_not_monotonic')
    return result


def _words(raw, variables):
    text = re.sub(r'\([^)]*\)', '', raw).split(';', 1)[0].strip().upper()
    assignment = re.fullmatch(rf'#(\d+)\s*=\s*({NUMBER})', text)
    if assignment:
        variables[assignment[1]] = float(assignment[2])
        return {}, set(), set()
    def replace(match):
        if match[1] not in variables:
            raise ValueError('unknown_macro_variable_in_timeline')
        return str(variables[match[1]])
    text = re.sub(r'#(\d+)', replace, text)
    pairs = re.findall(WORD, text)
    return ({k: float(v) for k, v in pairs if k not in ('G', 'M', 'N', 'O')},
            {float(v) for k, v in pairs if k == 'G'},
            {int(float(v)) for k, v in pairs if k == 'M'})


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


def _duration(policy, key, assumptions, line):
    value = policy.get(key)
    if value is None:
        assumptions.append(dict(line=line, kind=key, assumption='zero_duration_not_measured'))
        return 0.
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(key + ' must be finite and nonnegative')
    return value


def prepare_source_timeline(raw_lines, segments, source, context):
    """Prepare base-only tensors on the complete source order.

    Estimated timing is a declared approximation, not a controller emulator.
    It never changes NC geometry or uses timing to recover missing cutting data.
    """
    supplied = context.get('prediction_timeline')
    if supplied is not None:
        if supplied.get('point_mapping') != 'source_process_order':
            raise ValueError('Full-position timeline needs source_process_order')
        time = np.asarray(supplied['time_s'], dtype=np.float64)
        commands = np.asarray(supplied['commands_rpm'], dtype=np.float64)
        initial = float(supplied['initial_speed_rpm'])
        queries = np.asarray(supplied['point_time_s'], dtype=np.float64)
        if (time.ndim != 1 or len(time) < 3 or commands.shape != time.shape
                or queries.shape != (len(source),) or
                not np.isfinite(np.r_[time, commands, queries, initial]).all()
                or initial < 0 or (commands < 0).any()):
            raise ValueError('Invalid complete source timeline')
        step = np.diff(time)
        if (step <= 0).any() or not np.allclose(step, np.median(step), rtol=.001, atol=1e-9):
            raise ValueError('Timeline must be uniform and strictly increasing')
        if (queries < time[0]).any() or (queries > time[-1]).any() or (np.diff(queries) < -1e-9).any():
            raise ValueError('Source times must be chronological and inside timeline')
        return dict(time_s=time[None], commands_rpm=commands[None], initial_speed_rpm=np.array([initial]),
                    point_time_s=queries), dict(mode='supplied_source_timeline',
                    source=supplied.get('source', 'caller_supplied'), assumptions=[], events=[],
                    command_at_source_rpm=commands[np.searchsorted(time, queries, side='right')-1].tolist(),
                    stopped_before_first_start=((initial == 0) &
                        (queries < (time[np.flatnonzero(commands > 0)[0]] if (commands > 0).any() else np.inf))).tolist(),
                    timing_is_measured=supplied.get('source') == 'measured')

    policy = context.get('prediction_policy', {})
    dt = _positive(policy.get('sample_period_s', .02), 'sample_period_s')
    initial = float(policy.get('initial_speed_rpm', 0.))
    if not math.isfinite(initial) or initial < 0:
        raise ValueError('initial_speed_rpm must be finite and nonnegative')
    assumptions = []
    if 'initial_speed_rpm' not in policy:
        assumptions.append(dict(kind='initial_speed_rpm', assumption='assumed_stopped_at_program_start', value=0.))
    variables = {}
    parsed = [_words(raw, variables) for raw in raw_lines]
    feeds = [w['F'] for w, _, _ in parsed if w.get('F', 0) > 0]
    source_feeds = source.source_feed_mm_min.to_numpy(float)
    positive_feeds = source_feeds[np.isfinite(source_feeds) & (source_feeds > 0)]
    fallback_feed = max(feeds + positive_feeds.tolist(), default=0.)
    if 'rapid_feed_mm_min' in policy:
        rapid_feed = _positive(policy['rapid_feed_mm_min'], 'rapid_feed_mm_min')
    elif fallback_feed > 0:
        rapid_feed = fallback_feed
        assumptions.append(dict(kind='rapid_feed_mm_min', value=rapid_feed,
            assumption='maximum_positive_program_or_source_feed_used_as_rapid_estimate'))
    else:
        rapid_feed = None
    by_line = {int(seg.line): seg for seg in segments}
    groups = {int(k): v for k, v in source.groupby('line', sort=False)}
    cumulative = source.path_mm.to_numpy(float)
    source_starts = np.r_[0., cumulative[:-1]]
    queries = np.full(len(source), np.nan)
    source_commands = np.full(len(source), np.nan)
    stopped_before_start = np.zeros(len(source), dtype=bool)
    ever_started = initial > 0
    clock = 0.
    setting = initial
    command = initial
    running = initial > 0
    motion = 0
    feed = 0.
    events = [dict(line=0, time_s=0., commands_rpm=initial, kind='initial_state')]
    for line_no, (words, codes, mcodes) in enumerate(parsed, 1):
        if 4 in mcodes:
            raise ValueError(f'M4 reverse spindle is not qualified by this base component (line {line_no})')
        for code in (0, 1, 2, 3):
            if code in codes:
                motion = code
        if 'F' in words:
            feed = words['F']
        if 'S' in words:
            setting = words['S']
        if 3 in mcodes:
            running = True
            ever_started = True
        if 5 in mcodes or mcodes & {30, 99}:
            running = False
        updated_command = setting if running else 0.
        if updated_command < 0 or not math.isfinite(updated_command):
            raise ValueError('Invalid spindle command')
        starting_spindle = updated_command > 0 and updated_command != command
        if updated_command != command:
            command = updated_command
            events.append(dict(line=line_no, time_s=clock, commands_rpm=command,
                               kind='spindle_command'))
        rows = groups.get(line_no)
        segment = by_line.get(line_no)
        prefix_duration = _duration(policy, 'spindle_wait_s', assumptions, line_no) if starting_spindle else 0.
        duration = 0.
        fractions = None
        if segment is not None:
            speed = _positive(segment.feed_mm_min, 'cutting program feed')
            duration = segment.length * 60. / speed
            if rows is not None:
                fractions = rows.segment_mm.to_numpy(float) / segment.length
                unknown = ~np.isfinite(fractions)
                if unknown.any():
                    # Time only. Geometry remains missing and cutting stays blocked.
                    fractions[unknown] = np.linspace(1./len(rows), 1., len(rows))[unknown]
                    assumptions.append(dict(line=line_no, kind='point_time',
                        assumption='ordered_time_fraction_only_geometry_still_unresolved'))
        elif rows is not None:
            start = source_starts[rows.index[0]]
            local = rows.path_mm.to_numpy(float) - start
            length = float(local[-1])
            if length > 0:
                speed = rapid_feed if motion == 0 or codes & {28, 53, 53.1} else feed
                if speed is None or not math.isfinite(speed) or speed <= 0:
                    speed = fallback_feed
                speed = _positive(speed, 'estimated positioning feed')
                duration = length * 60. / speed
                fractions = local / length
                assumptions.append(dict(line=line_no, kind='positioning_duration',
                    assumption='source_distance_and_program_feed_estimate_unknown_machine_kinematics',
                    distance_mm=length, feed_mm_min=speed))
            else:
                fractions = np.ones(len(rows))
        elif (any(k in words for k in 'XYZ') or codes & {28, 53, 53.1}) and 4 not in codes:
            assumptions.append(dict(line=line_no, kind='unestimated_motion_duration',
                assumption='zero_duration_no_source_or_supported_geometry_machine_motion_not_known'))
        # Rotations and machine events are retained even if they have no process rows.
        if any(k in words for k in 'ABC') and not any(k in words for k in 'XYZ'):
            duration += _duration(policy, 'rotary_move_s', assumptions, line_no)
        if 6 in mcodes:
            duration += _duration(policy, 'tool_change_s', assumptions, line_no)
        if mcodes & {0, 1}:
            duration += _duration(policy, 'optional_stop_s', assumptions, line_no)
        if 4 in codes:
            if 'X' in words:
                duration += max(0., words['X'])
            elif 'P' in words:
                unit = policy.get('dwell_p_unit', 'milliseconds')
                if unit not in ('seconds', 'milliseconds'):
                    raise ValueError('dwell_p_unit must be seconds or milliseconds')
                duration += max(0., words['P']) / (1000. if unit == 'milliseconds' else 1.)
                if 'dwell_p_unit' not in policy:
                    assumptions.append(dict(line=line_no, kind='dwell_p_unit', assumption='milliseconds'))
        if rows is not None:
            if fractions is None:
                fractions = np.ones(len(rows))
            queries[rows.index] = clock + prefix_duration + np.clip(fractions, 0., 1.) * duration
            source_commands[rows.index] = command
            stopped_before_start[rows.index] = initial == 0 and not ever_started
        clock += prefix_duration + duration
        if mcodes & {30, 99}:
            break
    if not np.isfinite(queries).all() or (np.diff(queries) < -1e-8).any():
        raise ValueError('Cannot place every source row chronologically in the program')
    tail = float(policy.get('tail_duration_s', 5.))
    if not math.isfinite(tail) or tail < 0:
        raise ValueError('tail_duration_s must be finite and nonnegative')
    total = max(clock + tail, dt*2)
    count = int(math.ceil(total/dt)) + 1
    if count > int(policy.get('maximum_timeline_points', 2000000)):
        raise ValueError('Timeline exceeds configured maximum_timeline_points')
    time = np.arange(count, dtype=np.float64) * dt
    event_times = np.array([e['time_s'] for e in events])
    event_commands = np.array([e['commands_rpm'] for e in events])
    event_indices = np.searchsorted(event_times, time, side='right') - 1
    commands = event_commands[event_indices]
    for event in events:
        event['effective_sample_time_s'] = float(time[min(np.searchsorted(time, event['time_s']), len(time)-1)])
    metadata = dict(mode='command_time_estimate', timing_is_measured=False,
        sample_period_s=dt, maximum_command_quantization_delay_s=dt,
        duration_s=clock, assumptions=assumptions, events=events,
        command_at_source_rpm=source_commands.tolist(),
        stopped_before_first_start=stopped_before_start.tolist(),
        controller_model='programmed_feed_and_explicit_event_assumptions_not_controller_emulation')
    return dict(time_s=time[None], commands_rpm=commands[None], initial_speed_rpm=np.array([initial]),
                point_time_s=queries), metadata


def prepare_full_positions(nc_path, process_path, context):
    """Software-owned file preparation. No model calculation occurs here."""
    lines, segments, info, reasons, events = parse_research_nc(Path(nc_path))
    source = read_process(Path(process_path), lines)
    tool = context.get('tool', {})
    diameter = _positive(tool.get('diameter_mm', 10.), 'tool diameter')
    teeth = _positive(tool.get('flutes', 3), 'tool flutes')
    frame, source, geometry_audit = geometry_grid(segments, info, source, diameter, teeth)
    tensors, timing = prepare_source_timeline(lines, segments, source, context)
    return dict(source=source, grid=frame, segments=segments, reasons=reasons, timeline=tensors,
                timing=timing, geometry_audit=geometry_audit, nc_events=events,
                diameter_mm=diameter, teeth=teeth)


def _base_at_source(model, timeline, *, extrapolate):
    device = next(model.parameters()).device
    tensors = {k: torch.as_tensor(v, dtype=torch.float64, device=device)
               for k, v in timeline.items()}
    with torch.no_grad():
        power, speed, supported = model.predict_dynamic_base(
            tensors['time_s'], tensors['commands_rpm'], tensors['initial_speed_rpm'],
            extrapolate=extrapolate)
        query = tensors['point_time_s'][None]
        values = model.dynamic_base.sample_points(tensors['time_s'], power, supported, query)
    return values.detach().cpu().numpy()[0], dict(
        timeline_points=int(power.numel()), valid_timeline_points=int(supported.sum()),
        time_series_power_min_w=float(torch.nan_to_num(power, nan=0.).min()),
        time_series_power_max_w=float(torch.nan_to_num(power, nan=0.).max()))


def predict_prepared_positions(prepared, model, profile, *, extrapolate=True):
    """Model runtime reference: unchanged cutting inference plus independent base."""
    grid, execution = infer_grid(model, prepared['grid'], profile)
    result = restore_source(prepared['source'], grid, prepared['reasons'])
    result, exact_queries = _predict_missing_supported_source_points(prepared, result, model, profile)
    base, base_execution = _base_at_source(model, prepared['timeline'], extrapolate=extrapolate)
    return combine_source_power(prepared, result, base, execution, exact_queries, base_execution,
        curve_max=float(model.base_speeds_rpm.max()),
        domain_min=float(model.dynamic_base.startup_domain_rpm[0]), extrapolate=extrapolate)


def combine_source_power(prepared, result, base, execution, exact_queries, base_execution,
                         *, curve_max, domain_min, extrapolate, event_domain_max=None):
    """Restore independently returned base values without replacing missing cutting."""
    original_reason = result.missing_reason.fillna('').copy()
    old_base = result.base_power_w.to_numpy(float).copy()
    old_values = result[OUT_COLS].to_numpy(float).copy()
    commands = np.asarray(prepared['timing']['command_at_source_rpm'])
    # No contact gives zero cutting even if the geometry is unavailable.
    # Source S=0 alone is not sufficient: the NC command must also be zero.
    known_initial_stop = np.asarray(prepared['timing'].get('stopped_before_first_start',
                                    np.zeros(len(result), dtype=bool)), dtype=bool)
    no_contact = ((result.ap_mm == 0) | (result.ae_mm == 0) |
                  ((result.source_spindle_rpm == 0) & (commands == 0) & known_initial_stop)).to_numpy()
    cut = old_values[:, :3] - old_base[:, None]
    cut[:, 0] = result.cutting_power_w.to_numpy(float)
    cut[no_contact] = 0.
    total = cut + base[:, None]
    result[OUT_COLS[:3]] = total
    result['base_power_w'] = base
    result['cutting_power_w'] = cut[:, 0]
    result['original_missing_reason'] = original_reason
    result['point_time_s'] = prepared['timeline']['point_time_s']
    result['base_mode'] = 'dynamic_component_extrapolation_enabled' if extrapolate else 'dynamic_component_qualified_domain'
    result['time_mode'] = prepared['timing']['mode']
    result['base_outside_steady_calibration'] = commands > curve_max
    # A braking response can still depend on an out-of-domain previous speed
    # even after its new command is zero. Retain that history conservatively.
    timeline_commands = np.asarray(prepared['timeline']['commands_rpm'])[0]
    initial = float(np.asarray(prepared['timeline']['initial_speed_rpm'])[0])
    event_limit = curve_max if event_domain_max is None else min(curve_max, event_domain_max)
    outside = ((timeline_commands > event_limit) |
               ((timeline_commands > 0) & (timeline_commands < domain_min)))
    history = np.maximum.accumulate(outside)
    query_indices = np.searchsorted(prepared['timeline']['time_s'][0],
                                     prepared['timeline']['point_time_s'], side='right')-1
    initial_outside = initial > event_limit or (0 < initial < domain_min)
    result['base_is_extrapolation'] = history[query_indices] | initial_outside
    good = np.isfinite(result[OUT_COLS].to_numpy()).all(axis=1)
    result.loc[good, 'prediction_status'] = 'research_prediction'
    result.loc[good, 'missing_reason'] = ''
    result.loc[good & no_contact, 'mapping_method'] = 'independent_dynamic_base_no_material_contact'
    result.loc[~good & ~np.isfinite(base), 'missing_reason'] = 'dynamic_base_unavailable'
    result.loc[~good & ~np.isfinite(base), 'prediction_status'] = 'blocked_dynamic_base'
    result.loc[~good & np.isfinite(base), 'prediction_status'] = 'blocked_cutting_geometry'
    if good.any() and ((result.loc[good, 'total_lower_w'] > result.loc[good, 'total_power_w']) |
                       (result.loc[good, 'total_upper_w'] < result.loc[good, 'total_power_w'])).any():
        raise ValueError('Unordered model interval')
    metadata = dict(schema='afc.full_position_prediction.1',
        status='complete_research_prediction' if good.all() else 'blocked_source_geometry_or_dynamic_state',
        complete=bool(good.all()), source_rows=len(result), predicted_rows=int(good.sum()),
        missing_rows=int((~good).sum()), noncontact_rows=int(no_contact.sum()),
        predicted_noncontact_rows=int((no_contact & good).sum()),
        missing_reasons=result.loc[~good, 'missing_reason'].value_counts().to_dict(),
        timing=prepared['timing'], cutting_execution=execution, base_execution=base_execution,
        exact_source_query_inference=exact_queries,
        geometry_audit=prepared['geometry_audit'], source_position_order_unchanged=True,
        cutting_geometry_missing_never_replaced_by_zero=True,
        includes_base_uncertainty=False, interval_kind='cutting_q05_q95_plus_dynamic_base_point_estimate',
        production_qualification=False, training_updates=0)
    return result, metadata


def exact_source_query_grid(prepared, result):
    """Geometry-only second pass for legal source cells omitted by the coarse grid."""
    need = result.total_power_w.isna() & np.isfinite(result.segment_mm)
    need &= (result.ap_mm > 0) & (result.ae_mm > 0) & (result.source_spindle_rpm > 0)
    indices = result.index[need].tolist()
    if not indices or prepared['grid'].empty:
        return None, indices
    frame = prepared['grid'].copy()
    frame['__source_query_index'] = -1
    additions = []
    segments = {seg.line: seg for seg in prepared['segments']}
    for idx in indices:
        point = result.loc[idx]
        seg = segments.get(int(point.line))
        if seg is None or not 0 <= point.segment_mm <= seg.length + 1e-8:
            continue
        block = frame.loc[frame.line == int(point.line)]
        if block.empty:
            continue
        q = float(point.segment_mm)
        u = q / seg.length
        row = block.iloc[0].copy()
        row['segment_mm'] = q
        row['segment_fraction'] = u
        row['path_mm'] = row['nc_path_mm'] = float(block.iloc[0].nc_path_mm) + q
        row['ap_mm'], row['ae_mm'] = float(point.ap_mm), float(point.ae_mm)
        row['__source_query_index'] = idx
        xyz, tangent = seg.points(u), _tangent(seg, u)
        for j, axis in enumerate('xyz'):
            row[f'{axis}_mm'], row[f'direction_{axis}'] = xyz[j], tangent[j]
        exact = block.index[np.isclose(block.segment_mm.to_numpy(), q, atol=1e-10, rtol=0.)]
        if len(exact):
            frame.loc[exact[0]] = row
        else:
            additions.append(row)
    if additions:
        frame = pd.concat([frame, pd.DataFrame(additions)], ignore_index=True)
    frame = frame.sort_values(['line', 'segment_mm'], kind='stable').reset_index(drop=True)
    return frame, indices


def _predict_missing_supported_source_points(prepared, result, model, profile):
    """Infer finite-geometry source points omitted by the coarser NC grid.

    Original finite outputs are immutable. New nodes use their actual NC
    coordinates and original ap/ae; no source-space interpolation is used.
    The same prospective window planner is used on this augmented query pass.
    """
    frame, indices = exact_source_query_grid(prepared, result)
    if frame is None:
        return result, dict(requested=0, predicted=0, original_finite_outputs_unchanged=True)
    inferred, execution = infer_grid(model, frame, profile)
    output = result.copy()
    original_finite = np.isfinite(result[OUT_COLS].to_numpy())
    filled = []
    for _, query in inferred.loc[inferred.__source_query_index >= 0].iterrows():
        idx = int(query.__source_query_index)
        values = query[OUT_COLS].to_numpy(float)
        if not np.isfinite(values).all():
            continue
        output.loc[idx, OUT_COLS] = values
        output.loc[idx, 'mapping_method'] = 'exact_supported_source_point_model_inference'
        output.loc[idx, 'prediction_status'] = STATUS
        output.loc[idx, 'missing_reason'] = ''
        filled.append(idx)
    np.testing.assert_array_equal(result[OUT_COLS].to_numpy()[original_finite],
                                  output[OUT_COLS].to_numpy()[original_finite])
    return output, dict(requested=len(indices), predicted=len(filled), source_indices=filled,
        original_finite_outputs_unchanged=True, execution=execution,
        basis='actual_NC_source_position_and_original_ap_ae_with_same_window_policy',
        added_query_sampling_density_is_research_estimate=True)


def predict_full_positions(nc_path, process_path, context, model, profile):
    prepared = prepare_full_positions(nc_path, process_path, context)
    return predict_prepared_positions(prepared, model, profile,
        extrapolate=bool(context.get('prediction_policy', {}).get('extrapolate_base', True)))

def geometry_grid(segments, info, source, diameter, teeth=3):
    frames=[]; distance=0.; coordinate_audit=[]
    source=source.copy()
    source['segment_mm']=np.nan
    source['segment_start_mm']=np.nan
    source['source_cell_end_mm']=np.nan
    source['model_input_feed_mm_min']=np.nan
    source['model_input_spindle_rpm']=np.nan
    source['prediction_status']='missing_input'
    source['missing_reason']='no_supported_nc_segment'
    for i,seg in enumerate(segments):
        n=max(2,int(np.ceil(seg.length/.5))+1)
        u=np.linspace(0.,1.,n); xyz=seg.points(u); tangent=_tangent(seg,u)
        prev=segments[i-1] if i and segments[i-1].group==seg.group else None
        nxt=segments[i+1] if i+1<len(segments) and segments[i+1].group==seg.group else None
        before=_tangent(prev,1.) if prev else np.zeros(3)
        after=_tangent(nxt,0.) if nxt else np.zeros(3)
        frame=pd.DataFrame(dict(line=seg.line,execution_instance=0,sequence_id=seg.group,
            nc_path_mm=distance+u*seg.length,path_mm=distance+u*seg.length,
            segment_mm=u*seg.length,segment_fraction=u,block_length_mm=seg.length,
            motion=seg.motion,curvature_per_mm=_curvature(seg),
            entry_angle_rad=_angle(before,tangent[0]) if prev else 0.,
            exit_angle_rad=_angle(tangent[-1],after) if nxt else 0.,
            previous_block_length_mm=prev.length if prev else 0.,
            next_block_length_mm=nxt.length if nxt else 0.,
            previous_feed_program_mm_min=prev.feed_mm_min if prev else 0.,
            next_feed_program_mm_min=nxt.feed_mm_min if nxt else 0.,
            has_previous_block=float(prev is not None),has_next_block=float(nxt is not None),
            diameter_mm=diameter,teeth=float(teeth),feed_input_mm_min=seg.feed_mm_min,
            spindle_rpm=seg.spindle_rpm,ap_mm=np.nan,ae_mm=np.nan,
            a_deg=info[seg.line]['a_deg'],c_deg=info[seg.line]['c_deg'],
            local_tilted_frame=info[seg.line]['local_tilted_frame']))
        for j,axis in enumerate('xyz'):
            frame[f'{axis}_mm']=xyz[:,j];frame[f'direction_{axis}']=tangent[:,j]
        ix=source.index[source.line==seg.line]
        reason=None
        try:
            if not len(ix): raise ValueError('no_process_rows_for_nc_segment')
            if source.loc[ix,'source_line_occurrence'].max()!=0: raise ValueError('repeated_nc_execution_unresolved')
            radius=float(np.linalg.norm(seg.start[:2]-seg.center[:2])) if seg.center is not None else 0.
            coords=reconstruct_process_coordinates(length_mm=float(seg.length),point_count=len(ix),motion=int(seg.motion),
                radius_mm=radius,sweep_rad=float(seg.sweep or 0.),z_change_mm=float(seg.end[2]-seg.start[2]))
            compatibility=source_distance_compatible(source.loc[ix,'source_s_token'].tolist(),coords)
            if not compatibility['compatible']: raise ValueError('source_s_incompatible_with_local_nc_geometry')
            source.loc[ix,'segment_mm']=coords.positions_mm
            source.loc[ix,'segment_start_mm']=coords.cell_starts_mm
            source.loc[ix,'source_cell_end_mm']=coords.cell_ends_mm
            source.loc[ix,'model_input_feed_mm_min']=seg.feed_mm_min
            source.loc[ix,'model_input_spindle_rpm']=seg.spindle_rpm
            cells=source_cell_indices(frame.segment_mm.to_numpy(),coords)
            values=source.loc[ix,['ap_mm','ae_mm']].to_numpy()
            supported=cells>=0
            if (values<0).any(): raise ValueError('negative_source_ap_ae')
            # Existing source-cell support convention; ae is never clipped to D.
            mapped=np.full((n,2),np.nan)
            mapped[supported]=values[cells[supported]]
            # Original interior linear mapping, without the production ae<=D gate.
            # Do not interpolate across a contact-state transition.
            knots=np.asarray(coords.positions_mm)
            pos=frame.segment_mm.to_numpy()
            right=np.clip(np.searchsorted(knots,pos,side='left'),0,len(knots)-1)
            left=np.clip(right-1,0,len(knots)-1)
            exact=np.isclose(pos,knots[right],rtol=0.,atol=1e-10)
            inside=(pos>=knots[0])&(pos<=knots[-1])
            contact=(values>0).all(axis=1)
            interp=np.column_stack([np.interp(pos,knots,values[:,j]) for j in range(2)])
            use=inside & (exact | ((contact[left]==contact[right]) & ((interp>0).all(axis=1)==contact[right])))
            mapped[use]=interp[use]
            frame[['ap_mm','ae_mm']]=mapped
            source.loc[ix,'missing_reason']='awaiting_model_return'
            coordinate_audit.append(dict(line=seg.line,points=len(ix),rule=coords.rule,source_s_compatible=True))
        except ValueError as exc:
            reason=str(exc)
            source.loc[ix,'missing_reason']=reason
            coordinate_audit.append(dict(line=seg.line,points=len(ix),supported=False,reason=reason))
        frames.append(frame); distance+=seg.length
    return (pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()),source,coordinate_audit

def infer_grid(model,frame,profile):
    if frame.empty:
        return frame.copy(), dict(grid_points=0,predicted_grid_points=0,windows=0,batches=0)
    selected,indices,windows=prospective_residual_context(frame,profile['execution']['window_points'])
    if not len(selected):
        out=frame.copy()
        for col in OUT_COLS: out[col]=np.nan
        return out,dict(grid_points=len(frame),predicted_grid_points=0,windows=0,batches=0)
    raw=selected[IN_COLS].to_numpy(np.float32)
    output=np.full((len(frame),5),np.nan)
    batches=0
    for group in window_batches(windows,128,length=len,strategy=profile['execution']['residual_batch_strategy']):
        shape=(len(group),max(map(len,group)))
        loc=np.zeros(shape,dtype=np.int64);valid=np.zeros(shape,dtype=bool)
        for k,win in enumerate(group):loc[k,:len(win)]=win;valid[k,:len(win)]=True
        device=next(model.parameters()).device
        with torch.no_grad():power=model(torch.from_numpy(raw[loc]).to(device),torch.from_numpy(valid).to(device)).detach().cpu().numpy()
        if not np.isfinite(power[valid]).all(): raise ValueError('nonfinite_research_prediction')
        output[indices[loc[valid]]]=power[valid]
        batches+=1
    out=frame.copy()
    out.loc[indices,'sequence_id']=selected.sequence_id.to_numpy()
    for i,col in enumerate(OUT_COLS):out[col]=output[:,i]
    return out,dict(grid_points=len(frame),predicted_grid_points=int(np.isfinite(output[:,0]).sum()),windows=len(windows),batches=batches)

def restore_source(source,grid,reasons):
    result=source.copy()
    for col in OUT_COLS:result[col]=np.nan
    result['research_status']=STATUS
    result['teeth_assumed']=3
    result['mapping_method']='unavailable'
    for line,rows in result.groupby('line',sort=False):
        g=grid.loc[grid.line==line] if not grid.empty else pd.DataFrame()
        if g.empty:
            result.loc[rows.index,'missing_reason']=reasons.get(int(line),'no_supported_nc_segment')
            continue
        values=g[OUT_COLS].to_numpy()
        valid=np.isfinite(values).all(axis=1)
        grid_contact=((g.ap_mm>0)&(g.ae_mm>0)).to_numpy()
        sequence=g.sequence_id.to_numpy()
        if not valid.any():continue
        # Every source point must stay inside an adjacent finite grid interval.
        positions=g.segment_mm.to_numpy()
        query=rows.segment_mm.to_numpy()
        for row_index,q in zip(rows.index,query):
            if not np.isfinite(q):continue
            hi=int(np.searchsorted(positions,q,side='left'));hi=min(hi,len(positions)-1)
            lo=max(0,hi-1)
            if abs(positions[hi]-q)<1e-10:lo=hi
            source_contact=bool(result.loc[row_index,'ap_mm']>0 and result.loc[row_index,'ae_mm']>0)
            if (valid[lo] and valid[hi] and positions[lo]-1e-10<=q<=positions[hi]+1e-10
                    and sequence[lo]==sequence[hi] and grid_contact[lo]==source_contact and grid_contact[hi]==source_contact):
                weight=0. if lo==hi else (q-positions[lo])/(positions[hi]-positions[lo])
                result.loc[row_index,OUT_COLS]=values[lo]*(1-weight)+values[hi]*weight
                result.loc[row_index,'mapping_method']='exact_nc_grid_point' if lo==hi else 'within_sequence_linear_interpolation'
            else:
                start=result.loc[row_index,'segment_start_mm'];end=result.loc[row_index,'source_cell_end_mm']
                candidates=np.flatnonzero(valid & (positions>start) & (positions<=end) & (grid_contact==source_contact))
                if not len(candidates):
                    result.loc[row_index,'missing_reason']='no_same_contact_grid_support_in_source_cell';continue
                nearest=candidates[np.argmin(np.abs(positions[candidates]-q))]
                between=(positions>=min(positions[nearest],q))&(positions<=max(positions[nearest],q))
                if not valid[between].all() or not (sequence[between]==sequence[nearest]).all():
                    result.loc[row_index,'missing_reason']='grid_sequence_boundary_in_source_cell';continue
                result.loc[row_index,OUT_COLS]=values[nearest]
                result.loc[row_index,'mapping_method']='source_cell_nearest_supported_grid'
            result.loc[row_index,'prediction_status']=STATUS
            result.loc[row_index,'missing_reason']=''
    return result
