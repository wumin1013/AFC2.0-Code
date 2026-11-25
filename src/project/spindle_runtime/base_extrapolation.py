"""Opt-in research extrapolation of the frozen dynamic base event component.

No parameter is fitted and no model buffer is changed. Entirely in-domain
requests delegate to the original component. Out-of-domain start/stop events
scale the nearest boundary event in time, speed and dynamic excess power.
"""
from __future__ import annotations

import torch


def _curve(module, speed):
    """Same piecewise-linear steady curve as model.forward, without clipping."""
    knots = module.speeds_rpm
    values = module.powers_w
    index = torch.sum(speed[..., None] >= knots, dim=-1) - 1
    index = index.clamp(0, knots.numel() - 2)
    return values[index] + (speed - knots[index]) * (
        values[index + 1] - values[index]) / (knots[index + 1] - knots[index])


def _domain(module, kind):
    domain = getattr(module, kind + "_domain_rpm")
    return float(domain[0]), float(domain[1])


def _anchor(module, kind, level):
    lower, upper = _domain(module, kind)
    anchor = min(max(level, lower), upper)
    return anchor, level / anchor


def _profile(module, kind, level):
    anchor, ratio = _anchor(module, kind, level)
    value = module.speeds_rpm.new_tensor([anchor])
    zero = torch.zeros_like(value)
    origin, target = (zero, value) if kind == "startup" else (value, zero)
    phase, power, speed, supported = module._event_profiles(origin, target)
    if not bool(supported.all()):
        raise ValueError("boundary_event_not_supported_by_original_component")
    return phase, power, speed, ratio


def _validate(module, time_s, commands_rpm, initial_speed_rpm):
    if not all(isinstance(v, torch.Tensor) for v in (time_s, commands_rpm, initial_speed_rpm)):
        raise TypeError("dynamic inputs must be torch tensors")
    device = module.speeds_rpm.device
    if any(v.device != device for v in (time_s, commands_rpm, initial_speed_rpm)):
        raise ValueError("dynamic input devices must match the component device")
    time, commands, initial = (v.to(torch.float64) for v in (time_s, commands_rpm, initial_speed_rpm))
    if time.ndim != 2 or time.shape[1] < 3 or commands.shape != time.shape or initial.shape != time.shape[:1]:
        raise ValueError("expected time/commands [B,T>=3] and settled initial speed [B]")
    if not all(bool(torch.isfinite(v).all()) for v in (time, commands, initial)):
        raise ValueError("dynamic timeline and settled initial speeds must be finite")
    if bool((commands < 0).any()) or bool((initial < 0).any()):
        raise ValueError("dynamic commands and initial speeds must be nonnegative")
    delta = torch.diff(time, dim=1)
    # torch.median(dim=...) also returns indices and rejects strict CUDA
    # determinism. Stable sorting selects the identical lower median value.
    median = torch.sort(delta, dim=1, stable=True).values[:, (delta.shape[1] - 1) // 2].unsqueeze(1)
    if bool((delta <= 0).any()) or not bool((torch.abs(delta - median) <= 1e-9 + .001 * torch.abs(median)).all()):
        raise ValueError("dynamic timeline must be strictly increasing and uniform")
    return time, commands, initial


def _events(module, time, commands, initial):
    """Validate all transitions before returning any usable prediction."""
    batches = []
    for batch in range(time.shape[0]):
        changes = torch.cat((torch.ones_like(commands[batch, :1], dtype=torch.bool),
                             commands[batch, 1:] != commands[batch, :-1]))
        indices = torch.nonzero(changes, as_tuple=False).flatten().tolist()
        events, ready_at = [], None
        for event_number, index in enumerate(indices):
            origin = float(initial[batch]) if index == 0 else float(commands[batch, index - 1])
            target = float(commands[batch, index])
            onset = float(time[batch, index])
            end = indices[event_number + 1] if event_number + 1 < len(indices) else time.shape[1]
            plateau = index == 0 and origin == target
            if ready_at is not None and onset < ready_at:
                raise ValueError(f"interrupted_event_before_recovery: batch={batch}, sample={index}, "
                                 f"onset={onset}, previous_ready={ready_at}")
            if plateau:
                profile, ratio, kind, duration = None, 1.0, "initial_plateau", 0.0
            else:
                if origin == 0 and target > 0:
                    kind, level = "startup", target
                elif origin > 0 and target == 0:
                    kind, level = "braking", origin
                else:
                    raise ValueError(f"unsupported_nonzero_to_nonzero_transition: batch={batch}, "
                                     f"sample={index}, origin={origin}, target={target}")
                phase, power, speed, ratio = _profile(module, kind, level)
                profile = phase, power, speed
                duration = float(phase[0, -1]) * ratio
            ready_at = onset + duration
            events.append(dict(start=index, end=end, onset=onset, kind=kind, origin=origin,
                               target=target, profile=profile, ratio=ratio, duration=duration))
        batches.append(events)
    return batches


def describe_extrapolation(dynamic_module, commands_rpm, initial_speed_rpm):
    """JSON-ready provenance for one request; computation remains tensor-only.

    The caller validates the timeline separately. Metadata must accompany any
    publication of extrapolated results; it is not a qualification receipt.
    """
    module = dynamic_module
    commands = torch.as_tensor(commands_rpm).detach().to(device="cpu", dtype=torch.float64)
    initial = torch.as_tensor(initial_speed_rpm).detach().to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(commands).all()) or not bool(torch.isfinite(initial).all()):
        raise ValueError("metadata speeds must be finite")
    if bool((commands < 0).any()) or bool((initial < 0).any()):
        raise ValueError("metadata speeds must be nonnegative")
    positive = module.speeds_rpm[module.speeds_rpm > 0]
    low, high = float(positive.min()), float(positive.max())
    all_levels = sorted(set(commands.flatten().tolist() + initial.flatten().tolist()))
    outside = [v for v in all_levels if v > 0 and (v < low or v > high)]
    event_domains = {kind: list(_domain(module, kind)) for kind in ("startup", "braking")}
    # The real component has the same steady and event boundaries. Also check
    # event domains so a later component cannot accidentally bypass scaling.
    event_outside = [v for v in all_levels if v > 0 and any(
        v < bounds[0] or v > bounds[1] for bounds in event_domains.values())]
    levels = sorted(set(outside + event_outside))
    offsets = []
    for level in levels:
        for kind in ("startup", "braking"):
            phase, power, speed, ratio = _profile(module, kind, level)
            anchor, _ = _anchor(module, kind, level)
            endpoint = -1 if kind == "startup" else 0
            endpoint_speed = speed[0, endpoint]
            mismatch = ratio * float(power[0, endpoint] - _curve(module, endpoint_speed))
            offsets.append(dict(kind=kind, requested_rpm=level, anchor_rpm=anchor, ratio=ratio,
                                original_event_platform_minus_steady_w=float(mismatch / ratio),
                                scaled_initial_plateau_vs_event_plateau_difference_w=mismatch,
                                complete_event_duration_s=float(phase[0, -1]) * ratio))
    return dict(schema="afc.dynamic_base_extrapolation.1", policy="boundary_shape_time_speed_excess_power_scaling_v1",
        extrapolated=bool(levels), status="unvalidated_research_extrapolation" if levels else "original_component_domain",
        validated_at_extrapolated_speeds=False, training_updates=0, parameters_modified=False,
        steady_domain_rpm=[low, high], event_domains_rpm=event_domains, extrapolated_levels_rpm=levels,
        equation="P_ext(t)=S(r*v_anchor(t/r))+r*(P_anchor(t/r)-S(v_anchor(t/r)))",
        scale="r=requested_rpm/nearest_positive_boundary_rpm; time and speed scale by r; excess-power energy scales by r^2",
        initial_plateau="Uses original steady curve with boundary linear extrapolation; zero is stopped",
        event_plateau="Retains scaled original event/steady offset to preserve endpoint and domain-boundary continuity",
        scaled_initial_plateau_vs_event_plateau_difference=offsets,
        unsupported=["nonzero_to_different_nonzero_speed", "target_change_before_previous_event_recovery"],
        uncertainty="No newly validated error bound or base uncertainty interval")


def predict_dynamic_base_extrapolated(dynamic_module, time_s, commands_rpm, initial_speed_rpm):
    """Return float64 power/speed and bool valid on the input tensor device.

    Valid describes supported computation under the declared research policy,
    not experimental qualification. Metadata identifies extrapolated requests.
    Unsupported transitions raise with their batch/sample and cause.
    """
    module = dynamic_module
    time, commands, initial = _validate(module, time_s, commands_rpm, initial_speed_rpm)
    events = _events(module, time, commands, initial)
    metadata = describe_extrapolation(module, commands, initial)
    if not metadata["extrapolated"]:
        return module(time_s, commands_rpm, initial_speed_rpm)
    power = torch.empty_like(time)
    speed = torch.empty_like(time)
    for batch, series in enumerate(events):
        for event in series:
            selected = slice(event["start"], event["end"])
            if event["kind"] == "initial_plateau":
                speed[batch, selected] = initial[batch]
                power[batch, selected] = _curve(module, initial[batch])
                continue
            phase, power_profile, speed_profile = event["profile"]
            ratio = event["ratio"]
            elapsed = (time[batch, selected] - event["onset"]) / ratio
            index = torch.zeros_like(elapsed, dtype=torch.long)
            anchor_power, anchor_speed = module._evaluate(elapsed, index, phase, power_profile, speed_profile)
            if ratio == 1.0:
                value, velocity = anchor_power, anchor_speed
            else:
                velocity = ratio * anchor_speed
                value = _curve(module, velocity) + ratio * (anchor_power - _curve(module, anchor_speed))
            power[batch, selected], speed[batch, selected] = value, velocity
    valid = torch.isfinite(power) & torch.isfinite(speed)
    if not bool(valid.all()):
        raise ValueError("nonfinite_extrapolated_event_computation")
    return power, speed, valid
