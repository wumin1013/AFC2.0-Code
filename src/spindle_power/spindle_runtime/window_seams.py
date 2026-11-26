"""Local correction of discontinuities introduced by residual inference windows."""
import numpy as np


def smooth_radial_engagement(radial_fraction, continuous):
    """Treat a moderate radial change as gradual when nearby slopes agree."""
    change = np.diff(radial_fraction)
    moderate = continuous & (np.abs(change) <= 0.05)
    directions = np.sign(change)
    support = np.zeros(len(change), dtype=np.int8)
    for distance in (1, 2):
        if len(change) <= distance:
            continue
        agree = moderate[:-distance] & moderate[distance:]
        if distance == 2:
            agree &= continuous[1:-1]
        agree &= (directions[:-distance] != 0) & (directions[:-distance] == directions[distance:])
        agree &= np.abs(change[:-distance] - change[distance:]) <= 0.02
        support[:-distance] += agree
        support[distance:] += agree
    return (np.abs(change) <= 0.02) | (support >= 2)


def cutting_seam_offsets(cutting, windows, continuous, smooth, predict_cutting, width):
    """Estimate a local scalar offset without changing geometry or interval widths."""
    half_span = 16
    offsets = np.zeros(len(cutting), dtype=np.float64)
    seams = []
    auxiliary_windows = 0
    for left, right in zip(windows[:-1], windows[1:]):
        boundary = int(right[0])
        seam = boundary - 1
        if left[-1] != seam or not continuous[seam] or not smooth[seam]:
            continue

        first = seam
        last = boundary
        while first > max(left[0], boundary - half_span) and continuous[first - 1] and smooth[first - 1]:
            first -= 1
        while last < min(right[-1], boundary + half_span - 1) and continuous[last] and smooth[last]:
            last += 1
        if min(boundary - first, last - seam) < 4:
            continue

        neighborhood = cutting[first:last + 1]
        jumps = np.abs(np.diff(neighborhood))
        local_step = float(np.median(np.delete(jumps, seam - first)))
        initial_gap = float(cutting[boundary] - cutting[seam])
        if abs(initial_gap) < max(60.0, 4.0 * local_step):
            continue

        # The two original windows are in the same known continuous sequence.
        aux_length = min(width, int(right[-1] - left[0] + 1))
        aux_start = int(np.clip(boundary - aux_length // 2, left[0], right[-1] - aux_length + 1))
        aux = predict_cutting(np.arange(aux_start, aux_start + aux_length))
        auxiliary_windows += 1
        aux_gap = float(aux[boundary - aux_start] - aux[seam - aux_start])
        if abs(aux_gap) > abs(initial_gap) * 0.5:
            continue

        # Keep the model's shared-context step at the seam and taper to the original values.
        left_weights = np.sin(np.linspace(0, np.pi / 2, boundary - first)) ** 2
        right_weights = np.cos(np.linspace(0, np.pi / 2, last - seam)) ** 2
        correction = np.r_[left_weights, -right_weights] * ((initial_gap - aux_gap) / 2)
        correction[[0, -1]] = 0
        local = slice(first, last + 1)
        corrected_max_step = float(np.max(np.abs(np.diff(neighborhood + correction))))
        if corrected_max_step > 0.75 * float(jumps.max()) or np.any(offsets[local]):
            continue

        offsets[local] = correction
        seams.append(dict(left=seam, right=boundary, first=first, last=last,
                          initial_cutting_step_w=initial_gap, corrected_cutting_step_w=aux_gap,
                          initial_max_local_step_w=float(jumps.max()),
                          corrected_max_local_step_w=corrected_max_step))
    return offsets, seams, auxiliary_windows
