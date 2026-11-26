"""Fixed-axis NC geometry and bounded-jerk autonomous feed estimates.

The planner is an estimate, not an emulation of a manufacturer's interpolator.
Spatial C2 splines are checked against NC geometry; a C2 quintic time law is
initialized by forward/backward acceleration propagation and conservatively
retimed against all three axis velocity/acceleration/jerk constraints.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly, CubicSpline

from .common import Cancel, checkpoint, version_of, save_json
from .measurement import detect_encoding

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
WORD = rf"([A-Z])\s*({NUMBER})"


@dataclass
class MachineConfig:
    name: str = "待配置机床"
    max_velocity_mm_min: list | None = None
    max_acceleration_mm_s2: list | None = None
    max_jerk_mm_s3: list | None = None
    interpolation_period_s: float | None = None
    contour_tolerance_mm: float | None = None
    corner_tolerance_mm: float | None = None
    initial_position_mm: list | None = None
    travel_min_mm: list | None = None
    travel_max_mm: list | None = None
    work_offsets_mm: dict = field(default_factory=dict)
    tool_lengths_mm: dict = field(default_factory=dict)
    home_position_mm: list | None = None
    fixed_rotary_degrees: dict = field(default_factory=dict)
    manufacturer_semantics: dict = field(default_factory=dict)
    spindle_wait_s: float | None = None
    tool_change_s: float | None = None
    optional_stop: str = "unknown"
    optional_stop_s: float | None = None
    lookahead: str = "full_program"
    provenance: str = "unconfigured"
    schema: str = "afc.machine.1"

    def save(self, path):
        save_json(path, self)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if data.get("schema") != "afc.machine.1":
            raise ValueError("机床配置格式不匹配")
        return cls(**data)

    def missing(self):
        errors = []
        for key in ("max_velocity_mm_min", "max_acceleration_mm_s2", "max_jerk_mm_s3"):
            value = getattr(self, key)
            if value is None or len(value) != 3 or not np.all(np.isfinite(value)) or min(value) <= 0:
                errors.append(f"{key} 需三个有限正数")
        for key in ("interpolation_period_s", "contour_tolerance_mm", "corner_tolerance_mm"):
            value = getattr(self, key)
            if value is None or not np.isfinite(value) or value <= 0:
                errors.append(f"{key} 需有限正数")
        for key in ("initial_position_mm", "travel_min_mm", "travel_max_mm"):
            value = getattr(self, key)
            if value is None or len(value) != 3 or not np.all(np.isfinite(value)):
                errors.append(f"缺少 {key}")
        if self.travel_min_mm is not None and self.travel_max_mm is not None:
            if np.any(np.asarray(self.travel_min_mm) >= self.travel_max_mm):
                errors.append("行程下限必须小于上限")
        if self.lookahead != "full_program":
            errors.append("首版仅支持 full_program 前瞻")
        for key in ("spindle_wait_s", "tool_change_s", "optional_stop_s"):
            value = getattr(self, key)
            if value is not None and (not np.isfinite(value) or value < 0):
                errors.append(f"{key} 必须非负有限")
        return errors


@dataclass
class Segment:
    line: int
    start: np.ndarray
    end: np.ndarray
    motion: int
    feed_mm_min: float
    spindle_rpm: float
    group: int
    center: np.ndarray | None = None
    sweep: float = 0.

    @property
    def length(self):
        if self.center is None:
            return float(np.linalg.norm(self.end - self.start))
        radius = np.linalg.norm(self.start[:2] - self.center[:2])
        return float(np.hypot(radius * self.sweep, self.end[2] - self.start[2]))

    def points(self, u):
        u = np.asarray(u)
        if self.center is None:
            return self.start + u[..., None] * (self.end - self.start)
        radius = np.linalg.norm(self.start[:2] - self.center[:2])
        angle = math.atan2(self.start[1] - self.center[1], self.start[0] - self.center[0]) + self.sweep * u
        return np.stack([self.center[0] + radius * np.cos(angle),
                         self.center[1] + radius * np.sin(angle),
                         self.start[2] + u * (self.end[2] - self.start[2])], axis=-1)


def _arc(start, end, words, clockwise):
    def sweep(center):
        a = math.atan2(start[1] - center[1], start[0] - center[0])
        b = math.atan2(end[1] - center[1], end[0] - center[0])
        delta = (a - b) % (2 * math.pi) if clockwise else (b - a) % (2 * math.pi)
        if delta < 1e-10:
            delta = 2 * math.pi
        return -delta if clockwise else delta
    if "R" in words:
        r = abs(words["R"])
        chord = end[:2] - start[:2]
        length = np.linalg.norm(chord)
        if length < 1e-10 or length > 2 * r + 1e-7:
            raise ValueError("R圆弧半径无效或整圆缺少圆心")
        normal = np.array([-chord[1], chord[0]]) / length
        h = np.sqrt(max(0., r * r - length * length / 4))
        centers = [(start[:2] + end[:2]) / 2 + sign * h * normal for sign in (-1, 1)]
        center = next((c for c in centers if (abs(sweep(c)) <= math.pi + 1e-7) == (words["R"] >= 0)), centers[0])
        center = np.r_[center, start[2]]
    else:
        if "I" not in words and "J" not in words:
            raise ValueError("圆弧缺少 I/J 或 R")
        center = start + [words.get("I", 0), words.get("J", 0), 0]
    r0, r1 = np.linalg.norm(start[:2] - center[:2]), np.linalg.norm(end[:2] - center[:2])
    if r0 <= 1e-9 or abs(r0 - r1) > max(.01, r0 * .001):
        raise ValueError("圆弧起终点半径不一致")
    return center, sweep(center)


def parse_nc(path, config=None, *, coordinate_mode="machine"):
    config = config or MachineConfig()
    if coordinate_mode not in ("machine", "workpiece_cutting"):
        raise ValueError("coordinate_mode必须为machine或workpiece_cutting")
    workpiece = coordinate_mode == "workpiece_cutting"
    text = Path(path).read_text(encoding=detect_encoding(path), errors="strict")
    position = np.asarray(config.initial_position_mm if config.initial_position_mm is not None and not workpiece else [np.nan] * 3, float)
    absolute, motion, feed, spindle_setting, spindle, group = True, 0, 0., 0., 0., 0
    offset = np.zeros(3)
    tool_length = 0.
    segments, issues, events, variables = [], [], [], {}
    controller_settings, active_wcs = [], None
    version = version_of(text)
    raw_lines = text.splitlines()
    logging_lines = set()
    # Recognize only this observed, side-effect-free-for-motion diagnostic block.
    # Never skip a generic IF that could hide a motion, variable write or jump.
    for i in range(len(raw_lines) - 2):
        condition = raw_lines[i].strip().upper()
        if re.fullmatch(r"IF\s+\[#1380 NE 0\](?: OR \[#138[1-5] NE 0\]){5}", condition) and raw_lines[i + 1].strip().upper() == 'PRINT P="ZERO_OFFSET_LOG.A,ZERO_OFFSET_LOG.TXT"' and raw_lines[i + 2].strip().upper() == "ENDIF":
            logging_lines.update((i + 1, i + 2, i + 3))
    for line_no, raw in enumerate(text.splitlines(), 1):
        if line_no in logging_lines:
            continue
        line = re.sub(r"\([^)]*\)", "", raw).split(";", 1)[0].strip().upper()
        # HNC exports may start with a pure percent-number program header.
        # Restrict this to line 1: never hide executable text or mid-program input.
        numbered_header = line_no == 1 and re.fullmatch(r"%[0-9]+", line)
        if not line or line == "%" or numbered_header or re.fullmatch(r"[ON]\d+", line):
            continue
        try:
            assignment = re.fullmatch(rf"#(\d+)\s*=\s*({NUMBER})", line)
            if assignment:
                variables[assignment[1]] = float(assignment[2])
                continue
            def expand(match):
                if match[1] not in variables:
                    raise ValueError(f"未定义宏变量 #{match[1]}")
                return str(variables[match[1]])
            line = re.sub(r"#(\d+)", expand, line)
            pairs = re.findall(WORD, line)
            remainder = re.sub(WORD, "", line).strip()
            if remainder:
                raise ValueError(f"不支持的宏/语法：{remainder}")
            codes = [float(v) for k, v in pairs if k == "G"]
            mcodes = [int(float(v)) for k, v in pairs if k == "M"]
            words = {k: float(v) for k, v in pairs if k not in ("G", "M", "N", "O")}
            supported = {0, 1, 2, 3, 4, 5.1, 17, 21, 28, 40, 43, 43.4, 49, 53, 54, 55, 56, 57, 58, 59, 90, 91, 94}
            bad = set(codes) - supported
            if bad:
                raise ValueError(f"未支持 G 代码 {sorted(bad)}")
            if set(mcodes) - {0, 1, 3, 5, 6, 8, 9, 30}:
                raise ValueError(f"未支持 M 代码 {mcodes}")
            if set(words) - set("XYZACBIJKRFSPETHL"):
                raise ValueError(f"未知地址 {sorted(set(words) - set('XYZACBIJKRFSPETHL'))}")
            if 90 in codes:
                absolute = True
            if 91 in codes:
                absolute = False
            for g in codes:
                if g in (0, 1, 2, 3):
                    motion = int(g)
                if 54 <= g <= 59:
                    key = f"G{int(g)}"
                    if workpiece:
                        if active_wcs is not None and active_wcs != key:
                            position[:] = np.nan
                            group += 1
                        active_wcs = key
                        offset = np.zeros(3)  # definition of workpiece coordinates
                    elif key not in config.work_offsets_mm:
                        raise ValueError(f"缺少 {key} 工件坐标偏置")
                    else:
                        offset = np.asarray(config.work_offsets_mm[key], float)
                if g == 49:
                    tool_length = 0.
                if g in (43, 43.4):
                    if g == 43.4 and config.manufacturer_semantics.get("G43.4") != "fixed_axis_length_compensation":
                        raise ValueError("G43.4 需核验固定刀轴刀长补偿语义")
                    h = str(int(words.get("H", -1)))
                    if workpiece and g == 43:
                        tool_length = 0.  # programmed tool-tip geometry, not a claimed H value
                    elif h not in config.tool_lengths_mm:
                        raise ValueError(f"缺少刀长 H{h}")
                    else:
                        tool_length = float(config.tool_lengths_mm[h])
                if g == 5.1 and workpiece:
                    controller_settings.append(dict(line=line_no, directive=line, semantics="uninterpreted_execution_setting"))
                if g == 5.1 and not workpiece and config.manufacturer_semantics.get("G05.1") != "contour_tolerance":
                    raise ValueError("G05.1 厂家平滑语义待核验")
                if g == 5.1 and words.get("P", 0) != 0 and "E" in words and config.contour_tolerance_mm is not None and config.contour_tolerance_mm > words["E"]:
                    raise ValueError("机床轮廓容差大于 G05.1 E 指定容差")
            for axis in "ABC":
                fixed_from_nc = workpiece and axis in words and abs(words[axis]) <= 1e-8
                if axis in words and not fixed_from_nc and (axis not in config.fixed_rotary_degrees or abs(words[axis] - config.fixed_rotary_degrees[axis]) > 1e-8):
                    raise ValueError(f"{axis} 轴未确认为固定角度；首版不支持旋转联动")
            if "F" in words:
                feed = words["F"]
                if feed <= 0:
                    raise ValueError("F 必须为正数")
            if "S" in words:
                spindle_setting = words["S"]
                if spindle_setting < 0:
                    raise ValueError("S 必须非负")
                if spindle > 0:
                    spindle = spindle_setting
            if 3 in mcodes:
                spindle = spindle_setting
            if 5 in mcodes:
                spindle = 0.
            if any(m in mcodes for m in (0, 1, 3, 5, 6)) or "S" in words or 4 in codes:
                group += 1
                duration, reason = 0., "spindle"
                if 0 in mcodes or 1 in mcodes:
                    reason = "operator_stop"
                    duration = 0. if 1 in mcodes and config.optional_stop == "skip" else config.optional_stop_s
                elif 6 in mcodes:
                    duration, reason = config.tool_change_s, "tool_change"
                elif 4 in codes:
                    duration, reason = words.get("X", words.get("P")), "dwell"
                    if config.manufacturer_semantics.get("G04") != "seconds":
                        raise ValueError("G04 驻留单位未确认为秒")
                else:
                    duration = config.spindle_wait_s
                events.append(dict(line=line_no, before_group=group, spindle_rpm=spindle,
                                   duration_s=duration, reason=reason))
            xyz = any(a in words for a in "XYZ") and 4 not in codes
            if xyz:
                if workpiece and (28 in codes or 53 in codes):
                    # A machine-coordinate/home move cannot define a workpiece
                    # start. Wait for explicit absolute coordinates before any
                    # cutting segment; do not invent the rapid motion or time.
                    for i, axis in enumerate("XYZ"):
                        if axis in words:
                            position[i] = np.nan
                    group += 1
                    continue
                target = position.copy()
                for i, axis in enumerate("XYZ"):
                    if axis in words:
                        target[i] = words[axis] + (0 if 53 in codes else offset[i] + (tool_length if i == 2 else 0)) if absolute else position[i] + words[axis]
                if not np.all(np.isfinite(position)) or not np.all(np.isfinite(target)):
                    if not workpiece or motion != 0:
                        issues.append(dict(line=line_no, message="缺少初始位置，首段路径待配置"))
                else:
                    center, sweep = (None, 0.) if motion < 2 or 28 in codes else _arc(position, target, words, motion == 2)
                    segment = Segment(line_no, position.copy(), target.copy(), 0 if 28 in codes else motion, feed, spindle, group, center, sweep)
                    if segment.length > 1e-9:
                        if segment.motion and feed <= 0:
                            raise ValueError("切削运动缺少模态 F")
                        segments.append(segment)
                position = target
                if workpiece and motion == 0:
                    group += 1
                if 28 in codes:
                    if config.home_position_mm is None:
                        raise ValueError("G28 缺少机床参考点")
                    home = position.copy()
                    for i, axis in enumerate("XYZ"):
                        if axis in words:
                            home[i] = config.home_position_mm[i]
                    group += 1
                    if np.all(np.isfinite(position)) and np.linalg.norm(home - position) > 1e-9:
                        segments.append(Segment(line_no, position.copy(), home.copy(), 0, 0., spindle, group))
                    position = home
                    group += 1
            if 30 in mcodes:
                break
        except (ValueError, KeyError, IndexError) as exc:
            issues.append(dict(line=line_no, message=str(exc)))
    if workpiece:
        segments = [segment for segment in segments if segment.motion in (1, 2, 3)]
    return dict(program_version=version, segments=segments, issues=issues, events=events,
                coordinate_mode=coordinate_mode, controller_settings=controller_settings,
                non_motion_logging_lines=sorted(logging_lines),
                timing_complete=all(e["duration_s"] is not None for e in events))


@dataclass
class FeedPlan:
    frame: pd.DataFrame
    program_version: str
    machine_version: str
    timing_complete: bool
    report: dict

    def save(self, path):
        save_json(path, dict(schema="afc.feed.1", program_version=self.program_version,
                            machine_version=self.machine_version, timing_complete=self.timing_complete,
                            report=self.report, columns=self.frame.to_dict(orient="list")))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
        if data.pop("schema", None) != "afc.feed.1":
            raise ValueError("进给计划格式不匹配")
        return cls(frame=pd.DataFrame(data.pop("columns")), **data)


def _geometry(segments, tolerance, cancel):
    # Refine the interpolation support until bounded pointwise deviation from
    # the exact NC parametrization is below half the user contour allowance.
    lengths = np.array([s.length for s in segments])
    ends = np.r_[0, np.cumsum(lengths)]
    for refinement in range(9):
        checkpoint(cancel)
        ss, pp = [0.], [segments[0].start]
        step = min(.5, max(.02, np.sqrt(tolerance))) / 2 ** refinement
        for k, segment in enumerate(segments):
            count = max(2, math.ceil(segment.length / step))
            if segment.center is not None:
                radius = np.linalg.norm(segment.start[:2] - segment.center[:2])
                count = max(count, math.ceil(abs(segment.sweep) / max(.001, np.sqrt(2 * tolerance / radius))))
            u = np.linspace(0, 1, count + 1)[1:]
            ss.extend(ends[k] + segment.length * u)
            pp.extend(segment.points(u))
        ss, pp = np.asarray(ss), np.asarray(pp)
        if len(ss) > 1000000:
            raise ValueError("几何采样超过100万点，需分段处理程序")
        spline = CubicSpline(ss, pp, axis=0, bc_type="natural")
        probe = np.sort(np.concatenate([ss, *[ss[:-1] + f * np.diff(ss) for f in (.25, .5, .75)]]))
        index = np.minimum(np.searchsorted(ends[1:], probe, side="right"), len(segments) - 1)
        exact = np.empty((len(probe), 3))
        for k, segment in enumerate(segments):
            selected = index == k
            exact[selected] = segment.points((probe[selected] - ends[k]) / lengths[k])
        deviation = float(np.max(np.linalg.norm(spline(probe) - exact, axis=1)))
        if deviation <= tolerance * .5:
            return spline, ss, ends, deviation
    raise ValueError("路径平滑无法满足轮廓容差")


def _plan_group(segments, config, cancel):
    tolerance = min(config.contour_tolerance_mm, config.corner_tolerance_mm)
    spline, spatial_grid, ends, deviation = _geometry(segments, tolerance, cancel)
    # Geometry refinement must not introduce artificial acceleration resets at
    # every micrometre. Use a separate time-law support, retaining NC boundaries.
    grid = np.unique(np.r_[np.arange(0., ends[-1], .5), ends, (ends[:-1] + ends[1:]) / 2])
    grid = grid[np.r_[True, np.diff(grid) > 1e-8]]
    grid[-1] = ends[-1]
    segment_index = np.minimum(np.searchsorted(ends[1:], grid, side="right"), len(segments) - 1)
    vmax = np.array(config.max_velocity_mm_min) / 60
    amax, jmax = np.array(config.max_acceleration_mm_s2), np.array(config.max_jerk_mm_s3)
    tangent, curvature = spline(grid, 1), spline(grid, 2)
    programmed = np.array([segments[i].feed_mm_min / 60 if segments[i].motion else np.linalg.norm(vmax) for i in segment_index])
    # At feed boundaries, use the smaller of both adjacent limits.
    for i in range(1, len(segments)):
        j = np.searchsorted(grid, ends[i])
        previous = segments[i - 1].feed_mm_min / 60 if segments[i - 1].motion else np.linalg.norm(vmax)
        programmed[max(0, j - 1):j + 1] = np.minimum(programmed[max(0, j - 1):j + 1], previous)
    axis_cap = np.min(vmax / np.maximum(np.abs(tangent), 1e-12), axis=1)
    curve_cap = np.min(np.sqrt(amax / np.maximum(np.abs(curvature), 1e-12)), axis=1)
    jerk_cap = np.min(np.cbrt(jmax / np.maximum(np.abs(spline(grid, 3)), 1e-12)), axis=1)
    caps = np.minimum.reduce([programmed / np.maximum(np.linalg.norm(tangent, axis=1), 1e-12), axis_cap, curve_cap * .7, jerk_cap * .7])
    fine_tangent, fine_curve = spline(spatial_grid, 1), spline(spatial_grid, 2)
    fine_cap = np.minimum.reduce([
        np.min(vmax / np.maximum(np.abs(fine_tangent), 1e-12), axis=1),
        .7 * np.min(np.sqrt(amax / np.maximum(np.abs(fine_curve), 1e-12)), axis=1),
        .7 * np.min(np.cbrt(jmax / np.maximum(np.abs(spline(spatial_grid, 3)), 1e-12)), axis=1)])
    cells = np.clip(np.searchsorted(grid, spatial_grid, side="right") - 1, 0, len(grid) - 2)
    cell_limits = np.full(len(grid) - 1, np.inf)
    np.minimum.at(cell_limits, cells, fine_cap)
    caps[:-1] = np.minimum(caps[:-1], cell_limits)
    caps[1:] = np.minimum(caps[1:], cell_limits)
    v = caps.copy()
    v[0] = v[-1] = 0.
    conservative_a = float(amax.min()) * .25
    for i in range(1, len(v)):
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * conservative_a * (grid[i] - grid[i - 1])))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * conservative_a * (grid[i + 1] - grid[i])))
    scalar_jerk = float(jmax.min()) * .1 / max(1., float(np.max(np.abs(fine_tangent))))
    def jerk_limited_speed(low, high, distance):
        if high <= low or 1.5 * (high - low) * (high + low) ** 2 <= scalar_jerk * distance ** 2:
            return high
        left, right = low, high
        for _ in range(22):
            middle = (left + right) * .5
            if 1.5 * (middle - low) * (middle + low) ** 2 <= scalar_jerk * distance ** 2:
                left = middle
            else:
                right = middle
        return left
    for _ in range(3):
        checkpoint(cancel)
        for i in range(1, len(v)):
            v[i] = jerk_limited_speed(v[i - 1], v[i], grid[i] - grid[i - 1])
        for i in range(len(v) - 2, -1, -1):
            v[i] = jerk_limited_speed(v[i + 1], v[i], grid[i + 1] - grid[i])
    times = np.r_[0., np.cumsum(2 * np.diff(grid) / np.maximum(v[:-1] + v[1:], 1e-12))]
    def time_law(times, velocity):
        # Hermite endpoints (s,v,a=0), with dt=2*ds/(v0+v1), simplify
        # exactly to a quartic displacement / cubic S-shaped velocity law.
        # Vectorized coefficients avoid a Python solve per interpolation knot.
        dt, dv = np.diff(times), np.diff(velocity)
        return PPoly(np.vstack([-.5 * dv / dt**3, dv / dt**2,
                                np.zeros(len(dt)), velocity[:-1], grid[:-1]]), times)
    law = time_law(times, v)

    def evaluate(t):
        s = np.clip(law(t), grid[0], grid[-1])
        ds, dds, ddds = law(t, 1), law(t, 2), law(t, 3)
        r1, r2, r3 = spline(s, 1), spline(s, 2), spline(s, 3)
        vel = r1 * ds[:, None]
        accel = r2 * ds[:, None] ** 2 + r1 * dds[:, None]
        jerk = r3 * ds[:, None] ** 3 + 3 * r2 * (ds * dds)[:, None] + r1 * ddds[:, None]
        return s, spline(s), vel, accel, jerk


    # Include both sides of piece boundaries, where third derivatives may jump.
    probes = np.sort(np.concatenate([times, *[times[:-1] + f * np.diff(times) for f in (.001, .1, .25, .5, .75, .9, .999)]]))
    # Also inspect each refined geometry knot and its adjacent cell interior;
    # a short corner blend may fall between the coarse time-law probes.
    targets = np.sort(np.r_[spatial_grid, (spatial_grid[:-1] + spatial_grid[1:]) / 2])
    cells = np.clip(np.searchsorted(grid, targets, side="right") - 1, 0, len(grid) - 2)
    left, right = times[cells].copy(), times[cells + 1].copy()
    for _ in range(26):
        middle = (left + right) / 2
        below = law(middle) < targets
        left = np.where(below, middle, left)
        right = np.where(below, right, middle)
    probes = np.sort(np.r_[probes, (left + right) / 2])
    if np.min(law(probes, 1)) < -1e-7:
        raise ValueError("时标出现倒退，不能生成有效计划")
    s, _, vel, accel, jerk = evaluate(probes)
    idx = np.minimum(np.searchsorted(ends[1:], s, side="right"), len(segments) - 1)
    local_f = np.array([segments[i].feed_mm_min / 60 if segments[i].motion else np.linalg.norm(vmax) for i in idx])
    factors = {"axis_velocity": float(np.max(np.abs(vel) / vmax)),
               "axis_acceleration": float(np.sqrt(np.max(np.abs(accel) / amax))),
               "axis_jerk": float(np.cbrt(np.max(np.abs(jerk) / jmax))),
               "program_feed": float(np.max(np.linalg.norm(vel, axis=1) / local_f))}
    scale = max(1., *factors.values()) * 1.03
    duration = times[-1] * scale
    count = math.ceil(duration / config.interpolation_period_s)
    if count > 3000000:
        raise ValueError("计划超过300万采样点；请按强制停顿分段生成")
    output_time = np.r_[np.arange(count) * config.interpolation_period_s, duration]
    checkpoint(cancel)
    s, xyz, vel, accel, jerk = evaluate(output_time / scale)
    vel, accel, jerk = vel / scale, accel / scale ** 2, jerk / scale ** 3
    if (np.max(np.abs(vel) / vmax) > 1.001 or np.max(np.abs(accel) / amax) > 1.001 or np.max(np.abs(jerk) / jmax) > 1.001):
        raise ValueError("插补采样验证未满足运动约束，结果未保存")
    if np.any(xyz < np.array(config.travel_min_mm) - 1e-8) or np.any(xyz > np.array(config.travel_max_mm) + 1e-8):
        raise ValueError("轨迹超出机床行程")
    idx = np.minimum(np.searchsorted(ends[1:], s, side="right"), len(segments) - 1)
    r1, r2 = spline(s, 1), spline(s, 2)
    kappa = np.linalg.norm(np.cross(r1, r2), axis=1) / np.maximum(np.linalg.norm(r1, axis=1) ** 3, 1e-12)
    frame = pd.DataFrame(dict(time_s=output_time, path_mm=s, line=[segments[i].line for i in idx],
                              segment_mm=s - ends[idx], block_length_mm=np.diff(ends)[idx],
                              feed_cmd_mm_min=np.linalg.norm(vel, axis=1) * 60,
                              feed_program_mm_min=[segments[i].feed_mm_min for i in idx],
                              spindle_cmd_rpm=[segments[i].spindle_rpm for i in idx],
                              curvature_per_mm=kappa, motion=[segments[i].motion for i in idx],
                              execution_instance=0, limit_source=max(factors, key=factors.get)))
    for k, axis in enumerate("xyz"):
        frame[f"{axis}_mm"] = xyz[:, k]
        frame[f"v{axis}_mm_s"] = vel[:, k]
        frame[f"a{axis}_mm_s2"] = accel[:, k]
        frame[f"j{axis}_mm_s3"] = jerk[:, k]
    return frame, {"contour_deviation_mm": deviation, "retiming_scale": scale,
                   "max_velocity_ratio": float(np.max(np.abs(vel) / vmax)),
                   "max_acceleration_ratio": float(np.max(np.abs(accel) / amax)),
                   "max_jerk_ratio": float(np.max(np.abs(jerk) / jmax))}


def plan_feed(path, config: MachineConfig, *, cancel: Cancel = None):
    errors = config.missing()
    parsed = parse_nc(path, config)
    errors.extend(f"行 {i['line']}：{i['message']}" for i in parsed["issues"])
    if errors:
        raise ValueError("\n".join(errors))
    if not parsed["segments"]:
        raise ValueError("没有可规划的有效运动段")
    # Stops and changes between rapid/cutting modes delimit continuous paths.
    groups = []
    for seg in parsed["segments"]:
        if not groups or seg.group != groups[-1][-1].group or (seg.motion == 0) != (groups[-1][-1].motion == 0) or np.linalg.norm(seg.start - groups[-1][-1].end) > 1e-7:
            groups.append([])
        groups[-1].append(seg)
    frames, reports, elapsed, distance, used_events = [], [], 0., 0., set()
    for group in groups:
        checkpoint(cancel)
        for i, event in enumerate(parsed["events"]):
            if i not in used_events and event["before_group"] <= group[0].group:
                event["time_s"] = elapsed
                elapsed += event["duration_s"] or 0.
                used_events.add(i)
        frame, report = _plan_group(group, config, cancel)
        frame["time_s"] += elapsed
        frame["path_mm"] += distance
        # Distinguish a repeated physical line (e.g. G28 intermediate/home).
        frame["execution_instance"] = len(frames)
        elapsed, distance = float(frame.time_s.iloc[-1]), float(frame.path_mm.iloc[-1])
        frames.append(frame)
        reports.append(report)
    for i, event in enumerate(parsed["events"]):
        if i not in used_events:
            event["time_s"] = elapsed
            elapsed += event["duration_s"] or 0.
    return FeedPlan(pd.concat(frames, ignore_index=True), parsed["program_version"], version_of(asdict(config)),
                    parsed["timing_complete"], {"groups": reports, "events": parsed["events"],
                    "total_automatic_time_s": elapsed, "source": "autonomous_estimate", "provenance": config.provenance,
                    "timing": "complete_configured" if parsed["timing_complete"] else "relative; unknown pauses excluded"})
