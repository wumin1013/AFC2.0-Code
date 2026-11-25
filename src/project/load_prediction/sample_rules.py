"""Shared masks and sequence boundaries without a neural-runtime dependency."""
from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLE_RULES_VERSION = "afc.sample_rules.5"
QUALITY_FIELDS = ("tool_healthy", "no_alarm", "complete", "alignment_reliable")


def quality_pass(metadata):
    return all(metadata.get(key) is True for key in QUALITY_FIELDS)


def contact_mask(frame, metadata=None, *, require_known=True):
    """Contact comes from reviewed process evidence, never just a G command."""
    metadata = metadata or {}
    if {"ap_mm", "ae_mm"} <= set(frame):
        ap, ae = frame.ap_mm.to_numpy(float), frame.ae_mm.to_numpy(float)
        return np.isfinite(ap) & np.isfinite(ae) & (ap > 0) & (ae > 0)
    if "cutting_contact" in frame:
        return frame.cutting_contact.eq(True).fillna(False).to_numpy(bool)
    if metadata.get("cutting_contact_confirmed") is True:
        return np.ones(len(frame), bool)
    return np.zeros(len(frame), bool) if require_known else np.ones(len(frame), bool)


def motion_mask(frame):
    if "motion" in frame:
        return frame.motion.isin((1, 2, 3)).to_numpy(bool)
    if all(f"motion_g{i}" in frame for i in range(4)):
        return (frame[["motion_g1", "motion_g2", "motion_g3"]].sum(axis=1) > 0).to_numpy(bool)
    return np.zeros(len(frame), bool)


def apply_process_contact(frame, process):
    """Attach only known contact to an NC grid; independent of base power/tool D.

    Unknown support remains nullable rather than fabricated air or cutting.
    Used before both feed supervision and prospective feed/load prediction.
    """
    required = {"line", "execution_instance", "segment_mm", "ap_mm", "ae_mm"}
    if required - set(process):
        raise ValueError("工艺接触表缺少程序位置、切深或切宽")
    result = frame.reset_index(drop=True).copy()
    result["cutting_contact"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    for (line, instance), rows in result.groupby(["line", "execution_instance"], sort=False):
        source = process.loc[(process.line == line) & (process.execution_instance == instance)].sort_values("segment_mm")
        if len(source) < 2:
            continue
        positions = rows.segment_mm.to_numpy(float)
        if not np.isfinite(source.segment_mm.to_numpy(float)).all():
            continue
        # Rounded cumulative distances can repeat while ap/ae differ. Only the
        # contact statement may be merged when all rows at that position agree;
        # no geometric value is averaged and the process table remains intact.
        x = source.segment_mm.to_numpy(float)
        geometry = source[["ap_mm", "ae_mm"]].to_numpy(float)
        states = np.where(np.isfinite(geometry).all(axis=1) & (geometry >= 0).all(axis=1),
                          (geometry > 0).all(axis=1).astype(float), np.nan)
        starts = np.flatnonzero(np.r_[True, np.diff(x) > 0])
        low, high = np.minimum.reduceat(states, starts), np.maximum.reduceat(states, starts)
        x, state = x[starts], np.where(low == high, low, np.nan)
        right = np.searchsorted(x, positions, side="left")
        left = np.clip(right - 1, 0, len(x) - 1)
        right = np.clip(right, 0, len(x) - 1)
        exact = np.isclose(positions, x[right], rtol=0., atol=1e-10)
        inside = (positions >= x[0]) & (positions <= x[-1]) & np.isfinite(positions)
        same = np.isfinite(state[left]) & np.isfinite(state[right]) & (state[left] == state[right])
        known = inside & np.where(exact, np.isfinite(state[right]), same)
        result.loc[rows.index[known], "cutting_contact"] = state[right[known]].astype(bool)
    result["sequence_id"] = continuous_ids(result, known_only=True)
    return result


def continuous_ids(frame, valid=None, *, known_only=False):
    """Retain NC boundaries; every invalid point and execution change breaks.

    Observed gaps are supplied only while preparing supervised windows. A
    prospective call uses NC-known boundaries and never reads measured labels.
    """
    count = len(frame)
    if not count:
        return np.zeros(0, dtype=int)
    boundaries = np.zeros(count, bool)
    boundaries[0] = True
    for key in ("sequence_id", "execution_instance"):
        if key in frame:
            values = frame[key].to_numpy()
            boundaries[1:] |= values[1:] != values[:-1]
    if "motion" in frame:
        rapid = frame.motion.eq(0).to_numpy()
        boundaries[1:] |= rapid[1:] != rapid[:-1]
    if "cutting_contact" in frame or {"ap_mm", "ae_mm"} <= set(frame):
        contact = contact_mask(frame)
        boundaries[1:] |= contact[1:] != contact[:-1]
    for key in ("pause", "known_break"):
        if key in frame:
            stop = frame[key].eq(True).fillna(False).to_numpy(bool)
            boundaries[1:] |= stop[1:] | stop[:-1]
    if "path_mm" in frame:
        positions = frame.path_mm.to_numpy(float)
        delta = np.diff(positions)
        boundaries[1:] |= ~np.isfinite(delta) | (delta < -1e-8)
    if not known_only and "time_s" in frame:
        delta = np.diff(frame.time_s.to_numpy(float))
        positive = delta[np.isfinite(delta) & (delta > 0)]
        period = float(np.median(positive)) if len(positive) else np.inf
        boundaries[1:] |= ~np.isfinite(delta) | (delta <= 0) | (delta > 1.5 * period)
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != (count,):
            raise ValueError("连续段有效性遮罩长度不匹配")
        boundaries[1:] |= ~valid[1:] | ~valid[:-1]
    return np.cumsum(boundaries) - 1


def task_masks(frame, metadata):
    cutting = motion_mask(frame) & contact_mask(frame, metadata)
    feed = frame.get("feed_actual_mm_min", pd.Series(np.nan, index=frame.index)).to_numpy(float)
    valid_feed = cutting & quality_pass(metadata) & np.isfinite(feed) & (feed >= 0)
    if "aligned" in frame:
        valid_feed &= frame.aligned.eq(True).fillna(False).to_numpy(bool)
    power = frame.get("power_cut_w", pd.Series(np.nan, index=frame.index)).to_numpy(float)
    residual = valid_feed & np.isfinite(power)
    steady = frame.get("steady", pd.Series(False, index=frame.index)).eq(True).to_numpy(bool)
    stable = frame.get("spindle_stable", pd.Series(False, index=frame.index)).eq(True).to_numpy(bool)
    return {"valid_feed": valid_feed, "valid_residual": residual, "valid_main": residual & steady & stable,
            "cutting_contact": cutting}


def continuous_ranges(frame, valid=None, *, known_only=False):
    """Yield positional half-open ranges; never merge repeated sequence IDs."""
    use = np.ones(len(frame), bool) if valid is None else np.asarray(valid, bool)
    ids = continuous_ids(frame, use, known_only=known_only)
    boundaries = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1], True]) if len(ids) else []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if use[start:stop].all():
            yield int(start), int(stop)
