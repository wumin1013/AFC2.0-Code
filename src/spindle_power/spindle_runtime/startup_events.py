"""Isolated spindle startup events attached to program order."""

import numpy as np


def isolated_startup_inputs(target_rpm, elapsed_s):
    """Sample the calibrated response at arbitrary event-relative times."""
    end = np.where(elapsed_s > 0, elapsed_s, elapsed_s[-1])
    time = np.column_stack((np.zeros_like(end), end / 2, end))
    return dict(time_s=time, commands_rpm=np.full_like(time, target_rpm),
                initial_speed_rpm=np.zeros_like(end), point_time_s=elapsed_s[:, None])


def attach_startup_curves(prepared, metadata, profile, levels, phase_times, evaluate, *, unavailable_reason=None):
    """Publish event curves separately from the original process-point rows."""
    events = []
    profiles = {}
    domain = profile.get('base_event_calibration', {}).get('startup')
    for command in prepared.get('spindle_start_commands', ()):
        target = command['target_rpm']
        if not domain or not domain[0] <= target <= domain[-1]:
            events.append(dict(command, status='outside_startup_calibration'))
            continue
        if unavailable_reason is not None:
            events.append(dict(command, status=unavailable_reason))
            continue
        if target not in profiles:
            phases = [np.interp(target, levels, phase_times[:, i]) for i in range(phase_times.shape[1])]
            elapsed = np.concatenate([np.linspace(start, stop, 17)[:-1] for start, stop in zip(phases, phases[1:])]
                                     + [np.asarray(phases[-1:])])
            power = np.asarray(evaluate(target, elapsed), dtype=float)
            if power.shape != elapsed.shape or not np.isfinite(power).all():
                raise ValueError('Invalid isolated spindle startup power curve')
            peak = int(np.argmax(power))
            profiles[target] = dict(event_elapsed_s=elapsed.tolist(), base_power_w=power.tolist(),
                                    peak_w=float(power[peak]), peak_event_elapsed_s=float(elapsed[peak]),
                                    profile_duration_s=float(elapsed[-1]), cutting_power_w=0.)
        events.append(dict(command, status='calibrated_startup_curve', curve=profiles[target]))
    metadata['spindle_startup_events'] = events
