from __future__ import annotations

import time

from .segmentation import STATE_CODE_BY_TYPE
from .shared import *
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d


class PitModelMixin:
    _SEGMENTATION_PREDICTION_PROVENANCE_KEYS = (
        "segmentation_prediction_source",
        "segmentation_prediction_independent",
        "segmentation_temporary_measurement_mode",
        "segmentation_sample_prediction_context_signature",
        "segmentation_process_prediction_context_signature",
        "segmentation_process_prediction_source",
        "segmentation_process_prediction_row_count",
    )

    def _reset_smif_runtime_cache(self):
        self._smif_interaction_points = np.empty((0, 3), dtype=float)
        self._smif_bounds = None
        self._smif_focus_bounds = None
        self._smif_render_stats = {}
        self._smif_dashboard_payload = None
        self._smif_base_projection_payload = None
        self._smif_display_bounds = None

    def _get_smif_interval_line_width(self):
        return 1.8

    def _get_smif_interval_point_size(self):
        return 16.0

    def _get_smif_canvas_aspect_ratio(self):
        width_px = 0.0
        height_px = 0.0
        canvas = getattr(self, "canvas_smif", None)
        if canvas is not None:
            try:
                widget = canvas.get_tk_widget()
                width_px = float(widget.winfo_width())
                height_px = float(widget.winfo_height())
            except Exception:
                width_px = 0.0
                height_px = 0.0

        if width_px <= 1.0 or height_px <= 1.0:
            fig = getattr(self, "fig_smif", None)
            if fig is not None:
                try:
                    dpi = float(fig.get_dpi()) if fig.get_dpi() else 100.0
                    width_px = float(fig.get_figwidth()) * dpi
                    height_px = float(fig.get_figheight()) * dpi
                except Exception:
                    width_px = 0.0
                    height_px = 0.0

        if width_px <= 1.0 or height_px <= 1.0:
            return 1.0
        return max(float(width_px) / max(float(height_px), 1.0), 1.0)

    def _resolve_smif_display_transform(self, bounds=None):
        active_bounds = bounds if isinstance(bounds, tuple) and len(bounds) == 2 else self._get_smif_active_bounds()
        if not (isinstance(active_bounds, tuple) and len(active_bounds) == 2):
            return None
        try:
            mins = np.asarray(active_bounds[0], dtype=float)
            maxs = np.asarray(active_bounds[1], dtype=float)
        except Exception:
            return None
        if mins.shape != (3,) or maxs.shape != (3,):
            return None
        spans = np.asarray(maxs - mins, dtype=float)
        max_span = float(np.max(spans)) if spans.size else 1.0
        if not np.isfinite(max_span) or max_span <= 0.0:
            max_span = 1.0

        adjusted_spans = np.maximum(spans, max(0.45, max_span * 0.025))
        normalized_spans = adjusted_spans / max(max_span, 1.0)
        display_ratios = self._resolve_smif_display_aspect(normalized_spans)
        centers = (mins + maxs) / 2.0
        half_ranges = adjusted_spans / 2.0
        return {
            "mins": mins,
            "maxs": maxs,
            "centers": centers,
            "half_ranges": half_ranges,
            "display_ratios": np.asarray(display_ratios, dtype=float),
        }

    def _transform_smif_coords(self, coords, bounds=None):
        try:
            coord_arr = np.asarray(coords, dtype=float)
        except Exception:
            return np.empty((0, 3), dtype=float)
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return np.empty((0, 3), dtype=float)

        transform = self._resolve_smif_display_transform(bounds=bounds)
        if not isinstance(transform, dict):
            return np.asarray(coord_arr, dtype=float)

        centers = np.asarray(transform["centers"], dtype=float)
        half_ranges = np.asarray(transform["half_ranges"], dtype=float)
        display_ratios = np.asarray(transform["display_ratios"], dtype=float)
        safe_half_ranges = np.where(np.abs(half_ranges) > 1e-9, half_ranges, 1.0)
        transformed = ((coord_arr - centers) / safe_half_ranges) * display_ratios
        return np.asarray(transformed, dtype=float)

    def _transform_smif_axis_value(self, axis_idx, actual_value, bounds=None):
        transform = self._resolve_smif_display_transform(bounds=bounds)
        if not isinstance(transform, dict):
            return float(actual_value)
        try:
            axis = int(axis_idx)
            actual = float(actual_value)
            center = float(np.asarray(transform["centers"], dtype=float)[axis])
            half_range = float(np.asarray(transform["half_ranges"], dtype=float)[axis])
            display_ratio = float(np.asarray(transform["display_ratios"], dtype=float)[axis])
        except Exception:
            return float(actual_value)
        safe_half_range = half_range if abs(half_range) > 1e-9 else 1.0
        safe_display_ratio = display_ratio if abs(display_ratio) > 1e-9 else 1.0
        return float(((actual - center) / safe_half_range) * safe_display_ratio)

    def _inverse_smif_axis_value(self, axis_idx, display_value, bounds=None):
        transform = self._resolve_smif_display_transform(bounds=bounds)
        if not isinstance(transform, dict):
            return float(display_value)
        try:
            axis = int(axis_idx)
            display = float(display_value)
            center = float(np.asarray(transform["centers"], dtype=float)[axis])
            half_range = float(np.asarray(transform["half_ranges"], dtype=float)[axis])
            display_ratio = float(np.asarray(transform["display_ratios"], dtype=float)[axis])
        except Exception:
            return float(display_value)
        safe_display_ratio = display_ratio if abs(display_ratio) > 1e-9 else 1.0
        return float(center + display * half_range / safe_display_ratio)

    def _format_smif_axis_tick_value(self, actual_value):
        try:
            numeric = float(actual_value)
        except Exception:
            return str(actual_value)
        if not np.isfinite(numeric):
            return ""
        if abs(numeric) < 5e-9:
            numeric = 0.0
        text = f"{numeric:.2f}".rstrip("0").rstrip(".")
        return "0" if text in {"", "-0"} else text

    def _refresh_smif_main_axis_ticks(self, ax):
        if ax is None or not self._axis_is_3d(ax):
            return
        transform = self._resolve_smif_display_transform()
        if not isinstance(transform, dict):
            return

        try:
            actual_mins = np.asarray(transform["mins"], dtype=float)
            actual_maxs = np.asarray(transform["maxs"], dtype=float)
        except Exception:
            return
        if actual_mins.shape != (3,) or actual_maxs.shape != (3,):
            return

        axis_bindings = (
            (0, ax.get_xlim3d, ax.set_xticks, ax.set_xticklabels),
            (1, ax.get_ylim3d, ax.set_yticks, ax.set_yticklabels),
            (2, ax.get_zlim3d, ax.set_zticks, ax.set_zticklabels),
        )

        for axis_idx, get_limits, set_ticks, set_ticklabels in axis_bindings:
            try:
                display_limits = tuple(float(value) for value in get_limits())
            except Exception:
                continue
            if len(display_limits) != 2:
                continue

            bound_min = float(actual_mins[axis_idx])
            bound_max = float(actual_maxs[axis_idx])
            if not np.isfinite(bound_min) or not np.isfinite(bound_max):
                continue
            if bound_max < bound_min:
                bound_min, bound_max = bound_max, bound_min

            actual_span = float(bound_max - bound_min)
            if actual_span <= 1e-9:
                tick_values = np.asarray([bound_min], dtype=float)
            else:
                visible_min = self._inverse_smif_axis_value(axis_idx, min(display_limits))
                visible_max = self._inverse_smif_axis_value(axis_idx, max(display_limits))
                low = max(float(min(visible_min, visible_max)), bound_min)
                high = min(float(max(visible_min, visible_max)), bound_max)
                if high - low <= 1e-9:
                    low, high = bound_min, bound_max
                locator = MaxNLocator(nbins=5, min_n_ticks=3)
                tick_values = np.asarray(locator.tick_values(low, high), dtype=float)
                tol = max((high - low) * 1e-9, 1e-9)
                tick_values = tick_values[(tick_values >= low - tol) & (tick_values <= high + tol)]
                if tick_values.size == 0:
                    tick_values = np.asarray([low, high], dtype=float)
                tick_values = np.unique(np.clip(tick_values, bound_min, bound_max))

            display_ticks = np.asarray(
                [self._transform_smif_axis_value(axis_idx, value) for value in tick_values],
                dtype=float,
            )
            finite_mask = np.isfinite(display_ticks) & np.isfinite(tick_values)
            display_ticks = display_ticks[finite_mask]
            tick_values = tick_values[finite_mask]
            if display_ticks.size == 0:
                continue

            set_ticks(display_ticks.tolist())
            set_ticklabels([self._format_smif_axis_tick_value(value) for value in tick_values])

    def _set_smif_viewer_box(self, ax):
        if ax is None:
            return
        canvas_ratio = self._get_smif_canvas_aspect_ratio()
        width_ratio = min(max(canvas_ratio * 0.92, 1.35), 2.10)
        viewer_aspect = (width_ratio, 1.0, 0.92)
        viewer_zoom = min(max(1.16 + (canvas_ratio - 1.2) * 0.04, 1.14), 1.34)
        try:
            ax.set_box_aspect(viewer_aspect, zoom=viewer_zoom)
        except TypeError:
            ax.set_box_aspect(viewer_aspect)
        except Exception:
            pass

    def _resolve_smif_display_aspect(self, aspect_values):
        try:
            aspect = np.asarray(aspect_values, dtype=float).reshape(-1)
        except Exception:
            return np.asarray([1.0, 1.0, 1.0], dtype=float)
        if aspect.size != 3:
            return np.asarray([1.0, 1.0, 1.0], dtype=float)

        sanitized = []
        for value in aspect:
            try:
                numeric = float(value)
            except Exception:
                numeric = 1.0
            if not np.isfinite(numeric) or numeric <= 0.0:
                numeric = 1.0
            sanitized.append(max(numeric, SMIF_BOX_MIN_RATIO))
        sanitized = np.asarray(sanitized, dtype=float)

        max_value = float(np.max(sanitized)) if sanitized.size else 1.0
        if not np.isfinite(max_value) or max_value <= 0.0:
            max_value = 1.0
        normalized = sanitized / max_value

        view_mode = self._get_smif_view_mode()
        min_ratio = 0.58 if view_mode == "full" else 0.48
        softened = np.power(np.clip(normalized, SMIF_BOX_MIN_RATIO, None), 0.58)
        softened = np.maximum(softened, min_ratio)

        canvas_ratio = self._get_smif_canvas_aspect_ratio()
        if np.isfinite(canvas_ratio) and canvas_ratio > 1.25:
            horizontal_boost = min(1.20, 1.0 + (canvas_ratio - 1.25) * 0.10)
            depth_boost = min(1.12, 1.0 + (canvas_ratio - 1.25) * 0.05)
            softened[0] *= horizontal_boost
            softened[1] *= depth_boost

        return np.clip(softened, min_ratio, 1.35)

    def _resolve_smif_box_zoom(self, display_aspect):
        zoom = float(SMIF_BOX_ZOOM)
        try:
            aspect = np.asarray(display_aspect, dtype=float).reshape(-1)
        except Exception:
            aspect = np.asarray([1.0, 1.0, 1.0], dtype=float)
        if aspect.size != 3:
            aspect = np.asarray([1.0, 1.0, 1.0], dtype=float)

        min_ratio = float(np.min(aspect)) if aspect.size else 1.0
        if np.isfinite(min_ratio) and min_ratio < 0.80:
            zoom *= min(1.18, 1.0 + (0.80 - min_ratio) * 0.30)

        canvas_ratio = self._get_smif_canvas_aspect_ratio()
        if np.isfinite(canvas_ratio) and canvas_ratio > 1.25:
            zoom *= min(1.12, 1.0 + (canvas_ratio - 1.25) * 0.04)

        return min(max(zoom, 1.05), 1.42)

    def _set_smif_box_aspect(self, ax, aspect_values):
        if ax is None:
            return
        display_aspect = self._resolve_smif_display_aspect(aspect_values)
        zoom = self._resolve_smif_box_zoom(display_aspect)
        try:
            ax.set_box_aspect(tuple(float(val) for val in display_aspect), zoom=zoom)
        except TypeError:
            ax.set_box_aspect(tuple(float(val) for val in display_aspect))
        except Exception:
            pass

    def _get_smif_axis_limit_bounds(self, ax):
        if ax is None:
            return None
        try:
            mins = np.asarray(
                [
                    float(ax.get_xlim3d()[0]),
                    float(ax.get_ylim3d()[0]),
                    float(ax.get_zlim3d()[0]),
                ],
                dtype=float,
            )
            maxs = np.asarray(
                [
                    float(ax.get_xlim3d()[1]),
                    float(ax.get_ylim3d()[1]),
                    float(ax.get_zlim3d()[1]),
                ],
                dtype=float,
            )
        except Exception:
            return None
        if mins.shape != (3,) or maxs.shape != (3,):
            return None
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            return None
        if np.any(maxs <= mins):
            return None
        return mins, maxs

    def _resolve_smif_view_direction(self, elev, azim):
        try:
            elev_rad = math.radians(float(elev))
            azim_rad = math.radians(float(azim))
        except Exception:
            return np.asarray([0.0, -1.0, 0.0], dtype=float)

        direction = np.asarray(
            [
                math.cos(elev_rad) * math.cos(azim_rad),
                math.cos(elev_rad) * math.sin(azim_rad),
                math.sin(elev_rad),
            ],
            dtype=float,
        )
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 1e-9:
            return np.asarray([0.0, -1.0, 0.0], dtype=float)
        return direction / norm

    def _estimate_smif_projection_area(self, ax, elev, azim, bounds):
        del ax
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            return 0.0
        mins, maxs = bounds
        try:
            corners = np.asarray(
                [
                    [x_val, y_val, z_val]
                    for x_val in (mins[0], maxs[0])
                    for y_val in (mins[1], maxs[1])
                    for z_val in (mins[2], maxs[2])
                ],
                dtype=float,
            )
        except Exception:
            return 0.0
        if corners.shape != (8, 3) or not np.all(np.isfinite(corners)):
            return 0.0

        spans = np.maximum(np.asarray(maxs - mins, dtype=float), 1e-6)
        view_direction = self._resolve_smif_view_direction(elev, azim)
        primary_score = (
            float(spans[0] * spans[1] * abs(view_direction[2]))
            + float(spans[0] * spans[2] * abs(view_direction[1]))
            + float(spans[1] * spans[2] * abs(view_direction[0]))
        )

        world_up = np.asarray([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(view_direction, world_up))) > 0.96:
            world_up = np.asarray([0.0, 1.0, 0.0], dtype=float)
        try:
            screen_x = np.cross(world_up, view_direction)
            screen_x /= np.linalg.norm(screen_x)
            screen_y = np.cross(view_direction, screen_x)
            screen_y /= np.linalg.norm(screen_y)
        except Exception:
            return 0.0

        centered_corners = corners - np.mean(corners, axis=0)
        proj_x = centered_corners @ screen_x
        proj_y = centered_corners @ screen_y
        span_x = float(np.max(proj_x) - np.min(proj_x))
        span_y = float(np.max(proj_y) - np.min(proj_y))
        if not np.isfinite(span_x) or not np.isfinite(span_y):
            return 0.0
        larger_span = max(span_x, span_y, 1e-9)
        balance = min(span_x, span_y) / larger_span

        # 避免自动视角退化为几乎正侧视/正俯视的“薄片”效果。
        oblique_factor = max(0.15, 1.0 - float(np.max(np.abs(view_direction))))
        return primary_score * oblique_factor * (0.70 + 0.30 * balance)

    def _select_best_smif_view_angle(self, bounds):
        candidate_views = [
            (SMIF_MAIN_ELEV, SMIF_MAIN_AZIM),
            (18.0, -35.0),
            (28.0, -45.0),
            (18.0, -70.0),
            (18.0, -110.0),
            (28.0, -125.0),
            (24.0, -135.0),
            (14.0, -25.0),
            (14.0, -155.0),
        ]

        best_view = (float(SMIF_MAIN_ELEV), float(SMIF_MAIN_AZIM))
        best_score = -1.0
        for elev, azim in candidate_views:
            score = self._estimate_smif_projection_area(None, elev, azim, bounds)
            if score > best_score:
                best_score = score
                best_view = (float(elev), float(azim))
        return best_view

    def _auto_adjust_smif_view_angle(self, ax):
        bounds = self._get_smif_axis_limit_bounds(ax)
        if bounds is None:
            return
        best_view = self._select_best_smif_view_angle(bounds)

        try:
            ax.view_init(elev=best_view[0], azim=best_view[1])
        except Exception:
            pass

    def _axis_is_3d(self, ax):
        if ax is None:
            return False
        try:
            if str(getattr(ax, "name", "")).lower() == "3d":
                return True
        except Exception:
            pass
        return hasattr(ax, "add_collection3d") and hasattr(ax, "get_zlim3d")

    def _update_smif_bounds(self, coords):
        try:
            coord_arr = np.asarray(coords, dtype=float)
        except Exception:
            return
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return
        finite_mask = np.all(np.isfinite(coord_arr), axis=1)
        if not np.any(finite_mask):
            return
        finite_coords = coord_arr[finite_mask]
        mins = np.min(finite_coords, axis=0)
        maxs = np.max(finite_coords, axis=0)
        cached_bounds = getattr(self, "_smif_bounds", None)
        if isinstance(cached_bounds, tuple) and len(cached_bounds) == 2:
            try:
                cached_mins = np.asarray(cached_bounds[0], dtype=float)
                cached_maxs = np.asarray(cached_bounds[1], dtype=float)
                if cached_mins.shape == (3,) and cached_maxs.shape == (3,):
                    mins = np.minimum(mins, cached_mins)
                    maxs = np.maximum(maxs, cached_maxs)
            except Exception:
                pass
        self._smif_bounds = (mins, maxs)

    def _compute_smif_raw_bounds(self, coords):
        try:
            coord_arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return None
        finite_mask = np.all(np.isfinite(coord_arr), axis=1)
        if not np.any(finite_mask):
            return None
        finite_coords = coord_arr[finite_mask]
        return np.min(finite_coords, axis=0), np.max(finite_coords, axis=0)

    def _compute_smif_effective_bounds(self, coords):
        try:
            coord_arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return None

        finite_mask = np.all(np.isfinite(coord_arr), axis=1)
        if not np.any(finite_mask):
            return None
        finite_coords = coord_arr[finite_mask]
        if len(finite_coords) <= 24:
            return np.min(finite_coords, axis=0), np.max(finite_coords, axis=0)

        keep_threshold = max(48, int(math.ceil(len(finite_coords) * 0.80)))
        core_mask = np.ones(len(finite_coords), dtype=bool)
        trimmed = False

        for axis_idx in range(3):
            axis_vals = finite_coords[:, axis_idx]
            if axis_vals.size <= 24:
                continue
            if np.nanmax(axis_vals) - np.nanmin(axis_vals) <= 1e-9:
                continue

            q1, q3 = np.percentile(axis_vals, [25.0, 75.0])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr <= 1e-9:
                continue

            lower = float(q1 - 3.0 * iqr)
            upper = float(q3 + 3.0 * iqr)
            axis_mask = (axis_vals >= lower) & (axis_vals <= upper)
            if int(np.sum(axis_mask)) >= keep_threshold:
                core_mask &= axis_mask
                trimmed = True

        core_coords = finite_coords[core_mask] if trimmed else finite_coords
        if len(core_coords) < 12:
            core_coords = finite_coords
        return np.min(core_coords, axis=0), np.max(core_coords, axis=0)

    def _compute_smif_compact_bounds(self, coords, keep_ratio=0.60, min_keep_count=6):
        try:
            coord_arr = np.asarray(coords, dtype=float)
        except Exception:
            return None
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return None

        finite_mask = np.all(np.isfinite(coord_arr), axis=1)
        if not np.any(finite_mask):
            return None
        finite_coords = coord_arr[finite_mask]
        if len(finite_coords) <= 8:
            return np.min(finite_coords, axis=0), np.max(finite_coords, axis=0)

        keep_threshold = max(int(min_keep_count), int(math.ceil(len(finite_coords) * float(keep_ratio))))
        core_mask = np.ones(len(finite_coords), dtype=bool)
        trimmed = False

        for axis_idx in range(3):
            axis_vals = finite_coords[:, axis_idx]
            if axis_vals.size <= 8:
                continue
            if np.nanmax(axis_vals) - np.nanmin(axis_vals) <= 1e-9:
                continue

            q1, q3 = np.percentile(axis_vals, [25.0, 75.0])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr <= 1e-9:
                continue

            lower = float(q1 - 3.0 * iqr)
            upper = float(q3 + 3.0 * iqr)
            axis_mask = (axis_vals >= lower) & (axis_vals <= upper)
            if int(np.sum(axis_mask)) >= keep_threshold:
                core_mask &= axis_mask
                trimmed = True

        core_coords = finite_coords[core_mask] if trimmed else finite_coords
        if len(core_coords) < max(4, int(min_keep_count)):
            core_coords = finite_coords
        return np.min(core_coords, axis=0), np.max(core_coords, axis=0)

    def _resolve_smif_focus_bounds(self, coords, state_codes):
        try:
            coord_arr = np.asarray(coords, dtype=float)
            state_arr = np.asarray(state_codes, dtype=np.int8).reshape(-1)
        except Exception:
            return None
        if coord_arr.ndim != 2 or coord_arr.shape[1] != 3 or coord_arr.size == 0:
            return None
        if state_arr.size != len(coord_arr):
            return self._compute_smif_raw_bounds(coord_arr)

        for state_mask in (state_arr == 2, state_arr != 1):
            if int(np.sum(state_mask)) <= 0:
                continue
            bounds = self._compute_smif_raw_bounds(coord_arr[state_mask])
            if bounds is not None:
                return bounds
        return self._compute_smif_raw_bounds(coord_arr)

    def _get_smif_view_mode(self):
        view_mode_var = getattr(self, "smif_view_mode_var", None)
        mode = str(view_mode_var.get()).strip().lower() if view_mode_var is not None else "full"
        return "full" if mode == "full" else "focus"

    def _get_smif_scope_mode(self):
        scope_var = getattr(self, "smif_scope_var", None)
        mode = str(scope_var.get()).strip().lower() if scope_var is not None else "all"
        return "steady" if mode == "steady" else "all"

    def _extract_smif_segment_endpoints(self, segment):
        if not isinstance(segment, dict):
            return None
        try:
            start_point = np.asarray(
                [segment["start_x"], segment["start_y"], segment["start_z"]],
                dtype=float,
            )
            end_point = np.asarray(
                [segment["end_x"], segment["end_y"], segment["end_z"]],
                dtype=float,
            )
        except Exception:
            return None
        if not (np.all(np.isfinite(start_point)) and np.all(np.isfinite(end_point))):
            return None
        return start_point, end_point

    def _collect_smif_segment_points_array(self, segments):
        points = []
        for segment in segments or []:
            endpoints = self._extract_smif_segment_endpoints(segment)
            if endpoints is None:
                continue
            points.extend(endpoints)
        if not points:
            return np.empty((0, 3), dtype=float)
        return np.asarray(points, dtype=float)

    def _compute_smif_segment_bounds(self, segments, compact=False):
        points = self._collect_smif_segment_points_array(segments)
        if points.size == 0:
            return None
        if compact:
            return self._compute_smif_compact_bounds(points)
        return self._compute_smif_raw_bounds(points)

    def _expand_smif_bounds(self, bounds, ratio=0.10, min_margin=2.0):
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            return None
        try:
            mins = np.asarray(bounds[0], dtype=float)
            maxs = np.asarray(bounds[1], dtype=float)
        except Exception:
            return None
        if mins.shape != (3,) or maxs.shape != (3,):
            return None
        spans = np.maximum(maxs - mins, 0.0)
        max_span = float(np.max(spans)) if spans.size else 0.0
        if not np.isfinite(max_span) or max_span <= 0.0:
            max_span = 1.0
        base_margin = max(max_span * float(ratio), float(min_margin))
        margins = np.maximum(spans * float(ratio), base_margin)
        return mins - margins, maxs + margins

    def _segment_endpoints_within_smif_bounds(self, segment, bounds):
        endpoints = self._extract_smif_segment_endpoints(segment)
        if endpoints is None or not isinstance(bounds, tuple) or len(bounds) != 2:
            return False
        try:
            mins = np.asarray(bounds[0], dtype=float)
            maxs = np.asarray(bounds[1], dtype=float)
        except Exception:
            return False
        if mins.shape != (3,) or maxs.shape != (3,):
            return False
        return all(np.all(point >= mins) and np.all(point <= maxs) for point in endpoints)

    def _resolve_smif_trajectory_focus_segments(self, segments):
        usable_segments = [segment for segment in (segments or []) if self._extract_smif_segment_endpoints(segment) is not None]
        if not usable_segments:
            return [], None

        def _resolve_path_span(record):
            if not isinstance(record, dict):
                return None
            for start_key, end_key in (("start_s", "end_s"), ("path_start", "path_end")):
                try:
                    start_value = float(record.get(start_key))
                    end_value = float(record.get(end_key))
                except Exception:
                    continue
                if np.isfinite(start_value) and np.isfinite(end_value):
                    return (float(min(start_value, end_value)), float(max(start_value, end_value)))
            return None

        if self._get_smif_scope_mode() == "steady":
            steady_records = self._get_steady_interval_records()
            steady_spans = [span for span in (_resolve_path_span(record) for record in steady_records) if span is not None]
            if steady_spans:
                focus_segments = []
                for segment in usable_segments:
                    try:
                        seg_start = float(segment.get("path_start"))
                        seg_end = float(segment.get("path_end"))
                    except Exception:
                        continue
                    if not (np.isfinite(seg_start) and np.isfinite(seg_end)):
                        continue
                    seg_span = (float(min(seg_start, seg_end)), float(max(seg_start, seg_end)))
                    if any(seg_span[1] >= span[0] and seg_span[0] <= span[1] for span in steady_spans):
                        focus_segments.append(segment)
                if focus_segments:
                    focus_bounds = self._compute_smif_segment_bounds(focus_segments, compact=False)
                    return focus_segments, focus_bounds

        cutting_segments = [
            segment for segment in usable_segments
            if str(segment.get("motion_type") or "").strip().lower() == "cutting"
        ]
        if cutting_segments:
            cutting_bounds = self._compute_smif_segment_bounds(cutting_segments, compact=False)
            expanded_bounds = self._expand_smif_bounds(cutting_bounds, ratio=0.06, min_margin=1.2)
            if expanded_bounds is not None:
                nearby_segments = [
                    segment for segment in usable_segments
                    if self._segment_endpoints_within_smif_bounds(segment, expanded_bounds)
                ]
                if nearby_segments:
                    nearby_bounds = self._compute_smif_segment_bounds(nearby_segments, compact=False)
                    return nearby_segments, nearby_bounds or cutting_bounds
            return cutting_segments, cutting_bounds

        fallback_bounds = self._compute_smif_segment_bounds(usable_segments, compact=False)
        if fallback_bounds is None:
            return usable_segments, None
        expanded_bounds = self._expand_smif_bounds(fallback_bounds, ratio=0.05, min_margin=1.0)
        if expanded_bounds is None:
            return usable_segments, fallback_bounds
        filtered_segments = [
            segment for segment in usable_segments
            if self._segment_endpoints_within_smif_bounds(segment, expanded_bounds)
        ]
        if filtered_segments:
            filtered_bounds = self._compute_smif_segment_bounds(filtered_segments, compact=False)
            return filtered_segments, filtered_bounds or fallback_bounds
        return usable_segments, fallback_bounds

    def _get_smif_interval_records(self):
        return self._get_current_interval_records(allow_profile_fallback=False)

    def _get_smif_metric_label(self, metric_key):
        return "K_c^UCB" if str(metric_key) == "K_c_UCB" else "K_c_hat"

    def _build_smif_interval_annotations(self, row_indices, coords, interval_records, metric_key):
        row_arr = np.asarray(row_indices, dtype=int)
        coord_arr = np.asarray(coords, dtype=float)
        if row_arr.size == 0 or coord_arr.ndim != 2 or coord_arr.shape[1] != 3:
            return []

        usable_records = []
        for record in interval_records or []:
            if not isinstance(record, dict):
                continue
            if bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle":
                continue
            try:
                metric_value = float(record.get(metric_key))
            except Exception:
                continue
            if not np.isfinite(metric_value):
                continue
            usable_records.append((record, metric_value))
        if not usable_records:
            return []

        max_annotations = 18
        if len(usable_records) > max_annotations:
            stride = int(math.ceil(len(usable_records) / float(max_annotations)))
            usable_records = usable_records[::max(stride, 1)]

        bounds = getattr(self, "_smif_bounds", None)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            try:
                span = np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float)
            except Exception:
                span = np.ptp(coord_arr, axis=0)
        else:
            span = np.ptp(coord_arr, axis=0)
        span = np.asarray(span, dtype=float)
        max_span = float(np.max(span)) if span.size else 1.0
        if not np.isfinite(max_span) or max_span <= 0.0:
            max_span = 1.0
        z_lift = max(max_span * 0.035, 0.8)
        y_shift = max(max_span * 0.018, 0.5)
        x_shift = max(max_span * 0.010, 0.25)

        annotations = []
        metric_label = self._get_smif_metric_label(metric_key)
        for idx, (record, metric_value) in enumerate(usable_records):
            try:
                start_idx = int(record.get("start_idx"))
                end_idx = int(record.get("end_idx"))
            except Exception:
                bounds_map = self._resolve_interval_process_bounds(record)
                if not bounds_map:
                    continue
                start_idx = int(bounds_map.get("start_idx", -1))
                end_idx = int(bounds_map.get("end_idx", -1))
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            segment_mask = (row_arr >= start_idx) & (row_arr <= end_idx)
            if not np.any(segment_mask):
                continue
            segment_coords = coord_arr[segment_mask]
            mid_point = np.asarray(segment_coords[len(segment_coords) // 2], dtype=float)
            anchor_point = mid_point.copy()
            label_point = mid_point.copy()
            label_point[0] += ((idx % 3) - 1) * x_shift
            label_point[1] += ((idx % 2) * 2 - 1) * y_shift
            label_point[2] += z_lift * (1.0 + 0.15 * (idx % 3))
            zone_id = str(record.get("zone_id") or f"Z{idx + 1}")
            annotations.append({
                "anchor": anchor_point,
                "label_point": label_point,
                "text": f"{zone_id}\n{metric_label}={metric_value:.3f}",
            })
        return annotations

    def _draw_smif_interval_annotations(self, ax, row_indices, coords, interval_records, metric_key):
        annotations = self._build_smif_interval_annotations(row_indices, coords, interval_records, metric_key)
        if not annotations:
            return 0
        annotation_points = []
        for annotation in annotations:
            anchor_point = np.asarray(annotation["anchor"], dtype=float)
            label_point = np.asarray(annotation["label_point"], dtype=float)
            if not (np.all(np.isfinite(anchor_point)) and np.all(np.isfinite(label_point))):
                continue
            annotation_points.extend([anchor_point, label_point])
            try:
                ax.plot(
                    [anchor_point[0], label_point[0]],
                    [anchor_point[1], label_point[1]],
                    [anchor_point[2], label_point[2]],
                    color=SMIF_ANNOTATION_COLOR,
                    linewidth=0.9,
                    alpha=0.85,
                    zorder=8,
                )
            except Exception:
                pass
            try:
                ax.scatter(
                    [anchor_point[0]],
                    [anchor_point[1]],
                    [anchor_point[2]],
                    s=28,
                    c=SMIF_ANNOTATION_COLOR,
                    depthshade=False,
                    edgecolors="none",
                    zorder=9,
                )
            except Exception:
                pass
            try:
                ax.text(
                    label_point[0],
                    label_point[1],
                    label_point[2],
                    str(annotation["text"]),
                    color=SMIF_TEXT_COLOR,
                    fontsize=max(PLOT_FONT_BASE - 2, 9),
                    ha="center",
                    va="bottom",
                    zorder=10,
                    bbox=dict(
                        boxstyle="round,pad=0.28",
                        facecolor=(0.07, 0.12, 0.17, 0.72),
                        edgecolor=(1.0, 0.82, 0.40, 0.68),
                        linewidth=0.8,
                    ),
                )
            except Exception:
                pass
        if annotation_points:
            self._update_smif_bounds(np.asarray(annotation_points, dtype=float))
        return len(annotations)

    def _cancel_pending_smif_view_refresh(self):
        pending_job = getattr(self, "_smif_view_refresh_job", None)
        if pending_job is None:
            return
        try:
            self.root.after_cancel(pending_job)
        except Exception:
            pass
        self._smif_view_refresh_job = None

    def _capture_smif_view_state(self, ax):
        if ax is None:
            return None
        if not self._axis_is_3d(ax):
            try:
                return {
                    "mode": "2d",
                    "xlim": tuple(float(v) for v in ax.get_xlim()),
                    "ylim": tuple(float(v) for v in ax.get_ylim()),
                    "axis_pair": tuple(int(v) for v in getattr(self, "_smif_main_axis_pair", (0, 2))),
                }
            except Exception:
                return None
        try:
            return {
                "mode": "3d",
                "xlim": tuple(float(v) for v in ax.get_xlim3d()),
                "ylim": tuple(float(v) for v in ax.get_ylim3d()),
                "zlim": tuple(float(v) for v in ax.get_zlim3d()),
                "elev": float(getattr(ax, "elev", 24.0)),
                "azim": float(getattr(ax, "azim", -58.0)),
            }
        except Exception:
            return None

    def _apply_smif_view_state(self, ax, view_state):
        if ax is None or not isinstance(view_state, dict):
            return
        if not self._axis_is_3d(ax) or str(view_state.get("mode", "")).lower() == "2d":
            xlim = view_state.get("xlim")
            ylim = view_state.get("ylim")
            if xlim and len(xlim) == 2:
                try:
                    ax.set_xlim(float(xlim[0]), float(xlim[1]))
                except Exception:
                    pass
            if ylim and len(ylim) == 2:
                try:
                    ax.set_ylim(float(ylim[0]), float(ylim[1]))
                except Exception:
                    pass
            return
        try:
            elev = float(view_state.get("elev", 24.0))
            azim = float(view_state.get("azim", -58.0))
            ax.view_init(elev=elev, azim=azim)
        except Exception:
            pass

        limits = []
        for axis_key in ("xlim", "ylim", "zlim"):
            axis_limits = view_state.get(axis_key)
            if not axis_limits or len(axis_limits) != 2:
                return
            try:
                axis_min = float(axis_limits[0])
                axis_max = float(axis_limits[1])
            except Exception:
                return
            if axis_max <= axis_min:
                axis_max = axis_min + 1.0
            limits.append((axis_min, axis_max))

        ax.set_xlim3d(*limits[0])
        ax.set_ylim3d(*limits[1])
        ax.set_zlim3d(*limits[2])
        try:
            self._set_smif_viewer_box(ax)
        except Exception:
            pass
        ax.set_anchor('C')

    def _schedule_smif_view_refresh(self, delay_ms=120):
        ax = getattr(self, "ax_smif", None)
        if ax is None or not hasattr(self, "root"):
            return
        view_state = self._capture_smif_view_state(ax)
        if view_state is None:
            return
        self._cancel_pending_smif_view_refresh()

        def _run():
            self._smif_view_refresh_job = None
            try:
                self.refresh_smif_view(view_state=view_state, reuse_source_cache=True)
            except Exception:
                pass

        try:
            self._smif_view_refresh_job = self.root.after(int(max(delay_ms, 0)), _run)
        except Exception:
            pass

    def _cancel_pending_smif_refresh(self):
        pending_job = getattr(self, "_smif_refresh_job", None)
        if pending_job is None:
            return
        try:
            self.root.after_cancel(pending_job)
        except Exception:
            pass
        self._smif_refresh_job = None

    def _schedule_smif_refresh(self, delay_ms=0):
        if not hasattr(self, "root"):
            self.refresh_smif_view()
            return
        self._cancel_pending_smif_refresh()

        def _run():
            self._smif_refresh_job = None
            try:
                self.refresh_smif_view()
            except Exception:
                pass

        try:
            if int(delay_ms) > 0:
                self._smif_refresh_job = self.root.after(int(delay_ms), _run)
            else:
                self._smif_refresh_job = self.root.after_idle(_run)
        except Exception:
            self.refresh_smif_view()

    def refresh_mechanism_status_summary(self):
        def _safe_value(var, default=0.0):
            try:
                if hasattr(var, "get"):
                    value = var.get()
                else:
                    value = var
                value = float(value)
                return float(value) if np.isfinite(value) else float(default)
            except Exception:
                return float(default)

        idle_global = _safe_value(getattr(self, "p_idle_var", 0.0), 0.0)
        program_idle = _safe_value(getattr(self, "current_program_idle_power", 0.0), 0.0)
        idle_model = getattr(self, "idle_power_model", None)
        idle_speeds = []
        idle_powers = []
        if isinstance(idle_model, dict):
            try:
                idle_speeds = [float(v) for v in (idle_model.get("speeds") or []) if np.isfinite(float(v))]
                idle_powers = [float(v) for v in (idle_model.get("powers") or []) if np.isfinite(float(v))]
            except Exception:
                idle_speeds = []
                idle_powers = []

        if hasattr(self, "no_load_status_var"):
            if idle_speeds and idle_powers:
                idle_text = (
                    f"空载模型: {len(idle_speeds)} 点，转速范围 {min(idle_speeds):.1f}~{max(idle_speeds):.1f} rpm"
                )
                if program_idle > 0.0:
                    idle_text += f"，当前程序空载={program_idle:.3f} W"
                elif idle_global > 0.0:
                    idle_text += f"，全局P_idle={idle_global:.3f} W"
            elif idle_global > 0.0 or program_idle > 0.0:
                idle_text = "空载参数: "
                if idle_global > 0.0:
                    idle_text += f"全局P_idle={idle_global:.3f} W"
                if program_idle > 0.0:
                    if idle_global > 0.0:
                        idle_text += f"，当前程序空载={program_idle:.3f} W"
                    else:
                        idle_text += f"当前程序空载={program_idle:.3f} W"
                idle_text += "（未加载空载曲线模型）"
            else:
                idle_text = "空载功率未设定"
            self.no_load_status_var.set(idle_text)

        if hasattr(self, "step_feed_status_var"):
            kc_value = self._parse_optional_float(self.kc_coeff.get()) if hasattr(self, "kc_coeff") else None
            ke_value = self._parse_optional_float(self.ke_coeff.get()) if hasattr(self, "ke_coeff") else None
            kc_sigma = _safe_value(getattr(self, "kc_sigma", 0.0), 0.0)
            if kc_value is None and ke_value is None:
                step_text = "模型参数未设定"
            else:
                parts = []
                if kc_value is not None:
                    parts.append(f"K_c={float(kc_value):.6f}")
                if ke_value is not None:
                    parts.append(f"K_e={float(ke_value):.6f}")
                if kc_sigma > 0.0:
                    parts.append(f"σ_Kc={kc_sigma:.6f}")
                if hasattr(self, "has_posterior_curve_ready") and self.has_posterior_curve_ready():
                    parts.append("预测负载模型已就绪")
                step_text = "当前模型参数: " + "，".join(parts)
            self.step_feed_status_var.set(step_text)

        if hasattr(self, "kc_profile_status_var"):
            active_path = str(getattr(self, "active_kc_profile_path", "") or "").strip()
            active_profile = getattr(self, "active_kc_profile", None)
            has_profile_payload = self._profile_has_saved_payload(active_profile)
            profile_origin = self._get_profile_origin()
            if profile_origin == "runtime_identified_profile":
                profile_text = "案例配置: 当前运行时辨识结果（未保存）"
            elif active_path:
                profile_text = f"案例配置: {os.path.basename(active_path)}"
            elif has_profile_payload:
                profile_text = "案例配置: 当前内存配置（未保存）"
            elif idle_global > 0.0 or program_idle > 0.0 or self.has_identified_kc_ke():
                profile_text = "案例配置: 当前界面参数（未保存）"
            else:
                profile_text = "案例配置: 未加载"
            self.kc_profile_status_var.set(profile_text)

    def clear_kc_ke_state(self, persist=True, status_text="未辨识模型参数"):
        self.kc_coeff.set("")
        self.ke_coeff.set("")
        self.kc_sigma.set(0.0)
        self.step_feed_model_signature = ""
        self.imported_kc_profile = None
        self.imported_kc_profile_path = ""
        self.runtime_identified_kc_profile = None
        self.runtime_identified_profile_case_signature = ""
        self.active_kc_profile = None
        self.active_kc_profile_path = ""
        self.profile_origin = "no_profile"
        self.prediction_source = "no_profile"
        if self._has_authoritative_segmentation_state():
            preserved_intervals = self._get_current_interval_records(allow_profile_fallback=False)
            for record in preserved_intervals:
                for key in ("K_c_hat", "sigma_Kc", "K_c_UCB", "kc_hat", "sigma_kc"):
                    record.pop(key, None)
                if str(record.get("segment_type") or "").strip().lower() != "idle":
                    record["kc_source"] = ""
            self._set_current_interval_state(
                interval_records=preserved_intervals,
                segment_records=self._get_current_segment_records(allow_profile_fallback=False),
                point_kc_map={},
                source="segmentation",
                profile_locked=False,
                prediction_source="no_profile",
            )
        else:
            self._clear_current_interval_state()
        self._invalidate_process_alignment_caches(reason="clear_kc_ke_state")
        self.refresh_pit_button_state()
        self.refresh_smif_view()
        if hasattr(self, "step_feed_status_var"):
            self.step_feed_status_var.set(status_text)
        if hasattr(self, "prediction_mode_var"):
            self.prediction_mode_var.set("direct_prediction")
        self.refresh_mechanism_status_summary()
        self._sync_prediction_mode_after_model_change(prefer_posterior=False)
        if getattr(self, "manual_measurement_data", None):
            self._refresh_manual_measurement_prediction()
        if persist:
            self._persist_app_config()

    def _normalize_profile_origin(self, origin):
        normalized = str(origin or "").strip()
        if normalized in {"imported_profile", "runtime_identified_profile", "no_profile"}:
            return normalized
        return "no_profile"

    def _get_primary_input_file_or_empty(self):
        getter = getattr(self, "get_primary_input_file", None)
        if not callable(getter):
            return ""
        try:
            return str(getter() or "").strip()
        except Exception:
            return ""

    def _get_profile_origin(self):
        return self._normalize_profile_origin(getattr(self, "profile_origin", "no_profile"))

    def _get_prediction_source(self):
        return self._normalize_profile_origin(getattr(self, "prediction_source", self._get_profile_origin()))

    def _should_allow_imported_profile_autoload(self, allow_measurement_resolve=None):
        if bool(getattr(self, "_force_recompute_kc_profile", False)):
            return False
        if allow_measurement_resolve is True:
            return False
        return True

    def _clear_active_profile_state(self):
        self.active_kc_profile = None
        self.active_kc_profile_path = ""
        self.profile_origin = "no_profile"
        self.prediction_source = "no_profile"

    def _clear_imported_profile_state(self, clear_active=False, reason=""):
        had_profile = bool(getattr(self, "imported_kc_profile", None) or str(getattr(self, "imported_kc_profile_path", "") or "").strip())
        self.imported_kc_profile = None
        self.imported_kc_profile_path = ""
        if clear_active and self._get_profile_origin() == "imported_profile":
            self._clear_active_profile_state()
        if had_profile:
            self._debug_prediction_state_event(
                "clear_imported_profile",
                reason=str(reason or ""),
            )

    def _clear_runtime_identified_profile_state(self, clear_active=False, reason=""):
        had_profile = bool(getattr(self, "runtime_identified_kc_profile", None))
        self.runtime_identified_kc_profile = None
        self.runtime_identified_profile_case_signature = ""
        if clear_active and self._get_profile_origin() == "runtime_identified_profile":
            self._clear_active_profile_state()
        if had_profile:
            self._debug_prediction_state_event(
                "clear_runtime_profile",
                reason=str(reason or ""),
            )
        if str(reason or "") == "switch_to_sampledata":
            invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
            if callable(invalidator):
                invalidator(reason="切换实际负载文件")

    def _activate_profile_state(
        self,
        profile,
        *,
        origin="no_profile",
        file_path="",
        case_signature="",
        normalize=True,
    ):
        normalized_origin = self._normalize_profile_origin(origin)
        normalized_profile = (
            (
                self._normalize_loaded_kc_profile(
                    profile,
                    source_path=str(file_path or ""),
                )
                if normalize
                else dict(profile)
            )
            if isinstance(profile, dict)
            else None
        )
        if normalized_origin == "imported_profile":
            self.imported_kc_profile = dict(normalized_profile) if isinstance(normalized_profile, dict) else None
            self.imported_kc_profile_path = str(file_path or "")
        elif normalized_origin == "runtime_identified_profile":
            self.runtime_identified_kc_profile = dict(normalized_profile) if isinstance(normalized_profile, dict) else None
            self.runtime_identified_profile_case_signature = str(
                case_signature or self._get_current_measurement_case_signature()
            )
        self.active_kc_profile = dict(normalized_profile) if isinstance(normalized_profile, dict) else None
        self.active_kc_profile_path = str(file_path or "")
        self.profile_origin = normalized_origin if isinstance(normalized_profile, dict) else "no_profile"
        self.prediction_source = self.profile_origin
        return self.active_kc_profile

    def _build_profile_runtime_signature(self, profile=None, profile_path=""):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return ""
        template_context = self._resolve_profile_template_context(source_profile)
        point_count = len((source_profile or {}).get("point_kc_map", {}) or {})
        prediction_payload = {
            key: source_profile.get(key)
            for key in (
                "global_kc",
                "kc_sigma",
                "ke_value",
                "global_ke",
                "global_idle",
                "idle_power_model",
                "idle_model_signature",
                "line_kc_map",
                "point_kc_map",
                "sample_kc_profile",
            )
            if key in source_profile
        }
        content_signature = self._build_stable_prediction_digest(prediction_payload)
        return "|".join(
            [
                str(source_profile.get("source") or ""),
                str(point_count),
                str(template_context.get("process_hash") or ""),
                str(template_context.get("gcode_hash") or ""),
                self._normalize_profile_binding_path(profile_path or source_profile.get("profile_path")),
                content_signature,
            ]
        )

    @staticmethod
    def _build_stable_prediction_digest(payload):
        """对预测相关内容生成与字典插入顺序无关的稳定摘要。"""

        import hashlib

        def _normalize(value):
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, np.ndarray):
                return _normalize(value.tolist())
            if isinstance(value, dict):
                return {
                    str(key): _normalize(item)
                    for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
                }
            if isinstance(value, (list, tuple)):
                return [_normalize(item) for item in value]
            if isinstance(value, float):
                return float(value) if np.isfinite(value) else None
            if isinstance(value, (int, str, bool)) or value is None:
                return value
            return str(value)

        normalized = _normalize(payload)
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _build_current_prediction_model_signature(self):
        def _read_var(name, default=None):
            value = getattr(self, name, default)
            if hasattr(value, "get"):
                try:
                    value = value.get()
                except Exception:
                    value = default
            return value

        return self._build_stable_prediction_digest(
            {
                "idle_model_signature": str(getattr(self, "idle_model_signature", "") or ""),
                "idle_power_model": getattr(self, "idle_power_model", None),
                "step_feed_model_signature": str(
                    getattr(self, "step_feed_model_signature", "") or ""
                ),
                "global_kc": _read_var("kc_coeff"),
                "global_ke": _read_var("ke_coeff"),
                "global_idle": _read_var("p_idle_var"),
                "program_idle": _read_var("current_program_idle_power"),
            }
        )

    def _get_current_measurement_case_signature(self, measurement=None):
        payload = measurement if isinstance(measurement, dict) else getattr(self, "manual_measurement_data", None)
        if not isinstance(payload, dict):
            return ""
        binding = self._build_manual_measurement_binding(payload)
        if not binding:
            return ""
        try:
            import hashlib

            encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return hashlib.sha1(encoded).hexdigest()[:16]
        except Exception:
            return ""

    def _build_prediction_context_signature(self, prediction_source=None, measurement=None, process_path=None):
        normalized_prediction_source = self._normalize_profile_origin(
            prediction_source if prediction_source is not None else self._get_prediction_source()
        )
        current_process_path = str(process_path or self._get_primary_input_file_or_empty() or "").strip()
        process_signature = self._build_process_kc_profile_key(current_process_path) if current_process_path else ""
        measurement_signature = self._get_current_measurement_case_signature(measurement)
        profile_signature = ""
        if normalized_prediction_source == "imported_profile":
            profile_signature = self._build_profile_runtime_signature(
                getattr(self, "imported_kc_profile", None),
                profile_path=str(getattr(self, "imported_kc_profile_path", "") or ""),
            )
        elif normalized_prediction_source == "runtime_identified_profile":
            profile_signature = self._build_profile_runtime_signature(
                getattr(self, "runtime_identified_kc_profile", None),
                profile_path="",
            )
        return "|".join(
            [
                str(getattr(self, "sample_data_mode", "") or ""),
                process_signature,
                measurement_signature,
                normalized_prediction_source,
                profile_signature,
                str(getattr(self, "step_feed_model_signature", "") or ""),
                str(getattr(self, "idle_model_signature", "") or ""),
                self._build_current_prediction_model_signature(),
            ]
        )

    def _invalidate_process_alignment_caches(self, reason=""):
        self._process_point_lookup_cache = None
        self._process_point_lookup_cache_key = None
        self._process_point_metadata_cache_key = None
        self._sample_line_point_context_cache = None
        self._authoritative_segmentation_sample_lookup_cache = None
        self._segmentation_process_prediction_context_signature = ""
        self._segmentation_process_prediction_source = ""
        self._segmentation_process_prediction_row_count = 0
        if hasattr(self, "_smif_source_cache"):
            self._smif_source_cache = None
        self._process_model_state_version = int(getattr(self, "_process_model_state_version", 0) or 0) + 1
        if reason:
            self._debug_prediction_state_event(
                "invalidate_process_alignment_caches",
                reason=str(reason),
                kc_map_source="current_rows",
            )

    def _invalidate_segmentation_for_prediction_context_change(
        self,
        expected_context_signature,
        *,
        reason="预测模型发生变化",
    ):
        if not self._has_authoritative_segmentation_state():
            return False

        current_signature = str(
            getattr(self, "_current_interval_context_signature", "") or ""
        )
        expected_signature = str(expected_context_signature or "")
        if current_signature and expected_signature and current_signature == expected_signature:
            return False

        self._clear_current_interval_state(keep_profile_lock=False)
        self._latest_segmentation_result = None
        cleaner = getattr(self, "_clear_segmentation_output_artifacts", None)
        cleanup_error = ""
        if callable(cleaner):
            try:
                cleaner()
            except OSError as exc:
                cleanup_error = str(exc)
        if hasattr(self, "segmentation_status_var"):
            suffix = f"；旧导出清理失败（{cleanup_error}）" if cleanup_error else ""
            self.segmentation_status_var.set(f"全行程六类划分: 待重算（{reason}）{suffix}")
        self._debug_interval_state_event(
            "invalidate_segmentation_prediction_context",
            reason=str(reason or ""),
            previous_context_signature=current_signature or "none",
            expected_context_signature=expected_signature or "none",
            cleanup_error=cleanup_error or "none",
        )
        return True

    def _sync_measurement_case_state(self, measurement=None, reason=""):
        current_case_signature = self._get_current_measurement_case_signature(measurement)
        previous_case_signature = str(getattr(self, "measurement_case_signature", "") or "")
        self.measurement_case_signature = current_case_signature

        runtime_profile = self._resolve_runtime_identified_profile_for_current_case(
            measurement=measurement,
            process_path=self._get_primary_input_file_or_empty(),
        )
        if getattr(self, "runtime_identified_kc_profile", None) and not isinstance(runtime_profile, dict):
            self._clear_runtime_identified_profile_state(clear_active=True, reason=reason or "measurement_case_changed")

        if self._get_profile_origin() == "runtime_identified_profile" and not isinstance(runtime_profile, dict):
            self._clear_active_profile_state()
        elif self._get_profile_origin() == "runtime_identified_profile":
            self.prediction_source = "runtime_identified_profile"
        elif self._get_profile_origin() == "imported_profile":
            self.prediction_source = "imported_profile"
        else:
            self.prediction_source = "no_profile"

        self._last_process_application_context = ""
        self._debug_prediction_state_event(
            "sync_measurement_case_state",
            previous_case_signature=previous_case_signature or "none",
            measurement_case_signature=current_case_signature or "none",
            reason=str(reason or ""),
        )

    def _build_process_application_context_signature(self, program_name="", process_path=""):
        normalized_process_path = self._normalize_profile_binding_path(process_path or self._get_primary_input_file_or_empty())
        return "|".join(
            [
                str(getattr(self, "sample_data_mode", "") or ""),
                str(program_name or ""),
                normalized_process_path,
                str(getattr(self, "measurement_case_signature", "") or self._get_current_measurement_case_signature() or ""),
                self._get_prediction_source(),
            ]
        )

    def _can_reuse_current_interval_template(self, prediction_source=None, measurement=None):
        if not bool(getattr(self, "_current_interval_ready", False)):
            return False
        expected_prediction_source = self._normalize_profile_origin(
            prediction_source if prediction_source is not None else self._get_prediction_source()
        )
        current_prediction_source = self._normalize_profile_origin(
            getattr(self, "_current_interval_prediction_source", "no_profile")
        )
        if expected_prediction_source != current_prediction_source:
            return False
        expected_case_signature = self._get_current_measurement_case_signature(measurement)
        current_case_signature = str(getattr(self, "_current_interval_measurement_case_signature", "") or "")
        if expected_case_signature != current_case_signature:
            return False
        expected_context_signature = self._build_prediction_context_signature(
            prediction_source=expected_prediction_source,
            measurement=measurement,
        )
        current_context_signature = str(getattr(self, "_current_interval_context_signature", "") or "")
        return bool(expected_context_signature and expected_context_signature == current_context_signature)

    def _debug_prediction_state_event(self, event, **fields):
        payload = {
            "prediction_source": self._get_prediction_source(),
            "profile_origin": self._get_profile_origin(),
            "parameter_source": self._resolve_measurement_parameter_source(),
            "display_mode": self._get_measurement_display_mode(),
            "interval_source": str(getattr(self, "_current_interval_source", "") or "none"),
            "measurement_case_signature": str(
                fields.pop("measurement_case_signature", None)
                or getattr(self, "measurement_case_signature", "")
                or self._get_current_measurement_case_signature()
                or "none"
            ),
        }
        payload.update(fields)
        parts = [f"{key}={value}" for key, value in payload.items() if value not in (None, "")]
        message = f"[DEBUG][prediction-state] {event}"
        if parts:
            message = f"{message}: {', '.join(parts)}"
        try:
            print(message)
        except Exception:
            pass

    def _debug_interval_state_event(self, event, **fields):
        payload = {
            "interval_source": str(getattr(self, "_current_interval_source", "") or "none"),
            "profile_origin": self._get_profile_origin(),
            "prediction_source": self._get_prediction_source(),
            "display_mode": self._get_measurement_display_mode(),
        }
        payload.update(fields)
        parts = [f"{key}={value}" for key, value in payload.items()]
        message = f"[DEBUG][interval-state] {event}"
        if parts:
            message = f"{message}: {', '.join(parts)}"
        try:
            print(message)
        except Exception:
            pass

    def _clear_current_interval_state(self, keep_profile_lock: bool = False):
        previous_ready = bool(getattr(self, "_current_interval_ready", False))
        previous_source = str(getattr(self, "_current_interval_source", "") or "")
        previous_locked = bool(getattr(self, "_profile_intervals_locked", False))
        previous_context_signature = str(getattr(self, "_current_interval_context_signature", "") or "")
        previous_prediction_source = self._normalize_profile_origin(
            getattr(self, "_current_interval_prediction_source", "no_profile")
        )
        previous_case_signature = str(getattr(self, "_current_interval_measurement_case_signature", "") or "")
        empty_intervals = []
        self.current_interval_records = empty_intervals
        self.current_segment_records = []
        self.current_interval_point_kc_map = {}
        self._current_interval_ready = False
        self._current_interval_source = ""
        self._current_interval_context_signature = ""
        self._current_interval_prediction_source = "no_profile"
        self._current_interval_measurement_case_signature = ""
        self._authoritative_segmentation_sample_lookup_cache = None
        if not keep_profile_lock:
            self._profile_intervals_locked = False

        # 兼容旧变量，禁止其他位置直接写入
        self.pred_power_intervals = empty_intervals
        self.pit_records = empty_intervals
        self._cached_steady_intervals = {}
        if previous_ready or previous_source or previous_locked:
            self._debug_interval_state_event(
                "clear_current_state",
                previous_ready=previous_ready,
                previous_source=previous_source or "none",
                previous_profile_locked=previous_locked,
                previous_context_signature=previous_context_signature or "none",
                previous_prediction_source=previous_prediction_source,
                previous_case_signature=previous_case_signature or "none",
                keep_profile_lock=bool(keep_profile_lock),
            )

    def _set_current_interval_state(
        self,
        interval_records,
        *,
        segment_records=None,
        point_kc_map=None,
        source="",
        profile_locked=False,
        context_signature="",
        prediction_source="no_profile",
        measurement_case_signature="",
    ):
        if interval_records is not None and not isinstance(interval_records, list):
            raise TypeError("interval_records must be a list of dict records")
        if segment_records is not None and not isinstance(segment_records, list):
            raise TypeError("segment_records must be a list of dict records")
        if point_kc_map is not None and not isinstance(point_kc_map, dict):
            raise TypeError("point_kc_map must be a dict")

        source_text = str(source or "")
        has_authoritative_segmentation = bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )
        if has_authoritative_segmentation and source_text != "segmentation":
            self._debug_interval_state_event(
                "skip_non_authoritative_interval_write",
                attempted_source=source_text or "none",
                preserved_source="segmentation",
            )
            return False

        previous_source = str(getattr(self, "_current_interval_source", "") or "")
        previous_locked = bool(getattr(self, "_profile_intervals_locked", False))
        previous_ready = bool(getattr(self, "_current_interval_ready", False))
        normalized_intervals = [
            dict(record)
            for record in (interval_records or [])
            if isinstance(record, dict)
        ]
        normalized_segments = [
            dict(record)
            for record in (segment_records or [])
            if isinstance(record, dict)
        ]
        normalized_point_map = dict(point_kc_map or {})

        self.current_interval_records = normalized_intervals
        self.current_segment_records = normalized_segments
        self.current_interval_point_kc_map = normalized_point_map
        self._current_interval_ready = True
        locked_flag = bool(profile_locked)
        self._current_interval_source = source_text
        self._profile_intervals_locked = locked_flag
        self._current_interval_context_signature = str(context_signature or "")
        self._current_interval_prediction_source = self._normalize_profile_origin(prediction_source)
        self._current_interval_measurement_case_signature = str(
            measurement_case_signature or self._get_current_measurement_case_signature()
        )
        self._authoritative_segmentation_sample_lookup_cache = None

        # 兼容旧代码读取，禁止其他位置直接写入
        self.pred_power_intervals = self.current_interval_records
        self.pit_records = self.current_interval_records
        self._cached_steady_intervals = {
            "pit_records": [dict(record) for record in normalized_intervals],
            "segment_records": [dict(record) for record in normalized_segments],
            "point_kc_map": dict(normalized_point_map),
            "target_load_curve": list(getattr(self, "target_load_curve", []) or []),
        }
        self._debug_interval_state_event(
            "set_current_state",
            previous_ready=previous_ready,
            previous_source=previous_source or "none",
            previous_profile_locked=previous_locked,
            source=source_text or "none",
            profile_locked=locked_flag,
            context_signature=self._current_interval_context_signature or "none",
            prediction_source=self._current_interval_prediction_source,
            measurement_case_signature=self._current_interval_measurement_case_signature or "none",
            interval_count=len(normalized_intervals),
            segment_count=len(normalized_segments),
            point_kc_count=len(normalized_point_map),
        )
        return True

    def _sync_current_interval_state_prediction_context(self, prediction_source=None, measurement=None):
        if not bool(getattr(self, "_current_interval_ready", False)):
            return False

        resolved_prediction_source = self._normalize_profile_origin(
            prediction_source if prediction_source is not None else self._get_prediction_source()
        )
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        segment_records = self._get_current_segment_records(allow_profile_fallback=False)
        point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})
        self._set_current_interval_state(
            interval_records=interval_records,
            segment_records=segment_records,
            point_kc_map=point_kc_map,
            source=str(getattr(self, "_current_interval_source", "") or ""),
            profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
            context_signature=self._build_prediction_context_signature(
                prediction_source=resolved_prediction_source,
                measurement=measurement,
            ),
            prediction_source=resolved_prediction_source,
            measurement_case_signature=self._get_current_measurement_case_signature(measurement),
        )
        return True

    def _get_current_interval_records(self, allow_profile_fallback: bool = False):
        if bool(getattr(self, "_current_interval_ready", False)):
            # current state 一旦 ready，即使结果为空也必须尊重当前会话结果，不能静默回退旧 profile。
            return [dict(record) for record in (self.current_interval_records or []) if isinstance(record, dict)]
        if allow_profile_fallback:
            _origin, profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload(),
            )
            if isinstance(profile, dict):
                return self._extract_profile_interval_records(profile)
        return []

    def _get_current_segment_records(self, allow_profile_fallback: bool = False):
        if bool(getattr(self, "_current_interval_ready", False)):
            return [dict(record) for record in (self.current_segment_records or []) if isinstance(record, dict)]
        if allow_profile_fallback:
            _origin, profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload(),
            )
            if isinstance(profile, dict):
                return self._extract_profile_segment_records(profile)
        return []

    def _get_interval_sample_index_span(self, interval, line_numbers=None):
        if not isinstance(interval, dict):
            return None
        try:
            start_idx = int(interval.get("sample_start_idx"))
            end_idx = int(interval.get("sample_end_idx"))
        except Exception:
            start_idx = None
            end_idx = None
        if start_idx is not None and end_idx is not None:
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            return int(start_idx), int(end_idx)
        bounds = self._resolve_interval_sample_bounds(interval, line_numbers=line_numbers)
        if not bounds:
            return None
        return int(bounds["sample_start_idx"]), int(bounds["sample_end_idx"])

    def _invalidate_measurement_runtime_state(self, keep_profile_lock=True, clear_interval_state=True):
        preserve_segmentation = bool(
            clear_interval_state and self._has_authoritative_segmentation_state()
        )
        measurement = getattr(self, "manual_measurement_data", None)
        runtime_keys = (
            "mapped_ap",
            "mapped_ae",
            "mapped_feed",
            "mapped_mrr",
            "predicted_idle_power",
            "mapped_kc",
            "predicted_load",
            "line_no_aligned",
            "process_point_index",
            "process_point_count",
            "process_row_index",
            "process_point_anchor_x",
            "sample_anchor_x",
            "prediction_valid_mask",
            "cutting_load",
            "kc_point",
            "kc_valid_mask",
            "kc_gated_out_mask",
            "sample_kc_values",
            "sample_kc_valid_mask",
            "sample_kc_source",
            "idle_window_mask",
            "idle_point_mask",
            "process_anchor_mask",
            "prediction_updated_at",
            "sigma_idle",
            "delta_mrr",
            "gate_reference_kc",
            "measurement_binding",
            "measurement_runtime",
            "fit_df",
            "prediction_cache",
        ) + self._SEGMENTATION_PREDICTION_PROVENANCE_KEYS
        if isinstance(measurement, dict):
            for key in runtime_keys:
                measurement.pop(key, None)

        self._sample_line_point_context_cache = None
        self._smif_source_cache = None
        self._smif_dashboard_payload = None
        if clear_interval_state and not preserve_segmentation:
            self._clear_current_interval_state(
                keep_profile_lock=bool(keep_profile_lock and getattr(self, "_profile_intervals_locked", False))
            )
        elif preserve_segmentation:
            invalidator = getattr(self, "_invalidate_segmentation_sample_projection", None)
            if callable(invalidator):
                invalidator(reason="实际负载上下文变化")
            self._debug_interval_state_event("preserve_segmentation_on_measurement_invalidation")

    def _resolve_measurement_gate_reference_kc(self):
        measurement = getattr(self, "manual_measurement_data", None)
        if not isinstance(measurement, dict):
            return max(float(self.get_kc_value()), 1.0)

        try:
            gate_reference = float(measurement.get("gate_reference_kc", float("nan")))
        except Exception:
            gate_reference = float("nan")
        if np.isfinite(gate_reference) and gate_reference > 1e-12:
            return float(gate_reference)

        profile_origin, source_profile = self._resolve_forward_prediction_profile(
            measurement=measurement,
            process_path=self._get_primary_input_file_or_empty(),
            allow_autoload_imported=self._should_allow_imported_profile_autoload(),
        )
        try:
            profile_kc = float((source_profile or {}).get("global_kc", float("nan")))
        except Exception:
            profile_kc = float("nan")
        if np.isfinite(profile_kc) and profile_kc > 1e-12:
            gate_reference = float(profile_kc)
        else:
            gate_reference = max(float(self.get_kc_value()), 1.0)
        measurement["gate_reference_kc"] = float(gate_reference)
        self._debug_prediction_state_event(
            "resolve_measurement_gate_reference_kc",
            kc_map_source=profile_origin if profile_origin != "no_profile" else "current_rows",
        )
        return float(gate_reference)

    def _build_measurement_auto_identify_signature(self):
        process_path = str(self.get_primary_input_file() or "").strip()
        measurement_path = str(getattr(self, "manual_measurement_path", "") or "").strip()
        gcode_path = str(self.gcode_nc_path_var.get().strip()) if hasattr(getattr(self, "gcode_nc_path_var", None), "get") else ""
        if not process_path or not measurement_path:
            return ""

        parts = [
            self._normalize_profile_binding_path(gcode_path),
            self._normalize_profile_binding_path(process_path),
            self._normalize_profile_binding_path(measurement_path),
        ]
        for candidate_path in (process_path, measurement_path, gcode_path):
            try:
                mtime = os.path.getmtime(candidate_path) if candidate_path and os.path.exists(candidate_path) else 0.0
            except Exception:
                mtime = 0.0
            parts.append(f"{float(mtime):.6f}")
        return "|".join(parts)

    def _get_default_interval_policy(self):
        measurement_signature = self._get_current_measurement_case_signature()
        self.measurement_case_signature = measurement_signature
        allow_imported_autoload = self._should_allow_imported_profile_autoload()
        prediction_source, forward_profile = self._resolve_forward_prediction_profile(
            measurement=getattr(self, "manual_measurement_data", None),
            process_path=self._get_primary_input_file_or_empty(),
            allow_autoload_imported=allow_imported_autoload,
        )
        if prediction_source == "imported_profile" and self._profile_has_saved_payload(forward_profile):
            interval_policy = "use_active_profile"
        elif self._can_reuse_current_interval_template(
            prediction_source=prediction_source,
            measurement=getattr(self, "manual_measurement_data", None),
        ):
            interval_policy = "reuse_current_template"
        elif (
            getattr(self, "sample_data_mode", "") == "experiment_measurement"
            and getattr(self, "manual_measurement_data", None)
            and self.has_prediction_model_ready()
        ):
            interval_policy = "recompute_current"
        else:
            interval_policy = "fresh_or_empty"

        self._debug_prediction_state_event(
            "resolve_interval_policy",
            interval_policy=interval_policy,
            reused_current_template=bool(interval_policy == "reuse_current_template"),
            kc_map_source=prediction_source if prediction_source != "no_profile" else "current_rows",
        )
        return interval_policy

    def _sync_prediction_mode_after_model_change(self, prefer_posterior=False):
        if hasattr(self, "refresh_prediction_mode_controls"):
            self.refresh_prediction_mode_controls(prefer_posterior=prefer_posterior)
        if hasattr(self, "refresh_prediction_metrics_summary"):
            self.refresh_prediction_metrics_summary()

    def _arm_model_param_commit_refresh_suppression(self, duration_seconds=0.8):
        try:
            duration = max(float(duration_seconds or 0.0), 0.0)
        except Exception:
            duration = 0.8
        self._model_param_commit_refresh_suppressed_until = time.monotonic() + duration
        return None

    def _release_model_param_commit_refresh_suppression(self):
        self._model_param_commit_refresh_suppressed_until = 0.0
        return None

    def _is_model_param_commit_refresh_suppressed(self):
        try:
            suppressed_until = float(getattr(self, "_model_param_commit_refresh_suppressed_until", 0.0) or 0.0)
        except Exception:
            suppressed_until = 0.0
        return suppressed_until > time.monotonic()

    def _normalize_model_param_inputs_for_runtime(self):
        try:
            idle_power = float(self.p_idle_var.get())
        except Exception:
            idle_power = 0.0
            self.p_idle_var.set(idle_power)

        invalid_fields = []
        for label, var in (("K_c", self.kc_coeff), ("K_e", self.ke_coeff)):
            raw = str(var.get()).strip()
            if not raw:
                var.set("")
                continue
            numeric = self._parse_optional_float(raw)
            if numeric is None or numeric < 0:
                invalid_fields.append(label)
                var.set("")
            else:
                var.set(self._format_optional_model_param(numeric))

        self.current_program_idle_power.set(idle_power)
        self._update_program_idle_summary()
        self._update_manual_kcke_button_text()
        return idle_power, invalid_fields

    def _build_process_kc_profile_key(self, process_path=None):
        try:
            primary_input = self.get_primary_input_file()
        except Exception:
            primary_input = ""
        target_path = str(process_path or primary_input or "").strip()
        if not target_path:
            return ""
        normalized_path = os.path.normcase(os.path.abspath(target_path))
        try:
            process_mtime = os.path.getmtime(target_path)
        except Exception:
            process_mtime = 0.0
        return f"{normalized_path}|{process_mtime:.6f}"

    def _build_kc_profile_cache_path(self, profile_key, process_path=None):
        import hashlib

        key_text = str(profile_key or "").strip()
        if not key_text:
            return ""
        process_value = str(process_path or self.get_primary_input_file() or "").strip()
        process_stem = self._sanitize_kc_profile_stem(os.path.splitext(os.path.basename(process_value or "gcode_case"))[0])
        digest = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:24]
        cache_dir = str(getattr(self, "kc_profile_cache_dir", "") or "").strip()
        if not cache_dir:
            cache_dir = os.path.join(self.kc_profile_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{process_stem}_{digest}.kcke.json")

    def _strip_kc_profile_suffixes(self, file_name):
        stem = os.path.basename(str(file_name or "").strip()).lower()
        if stem.endswith(".json"):
            stem = stem[:-5]
        if stem.endswith(".kcke"):
            stem = stem[:-5]
        return stem.strip("._ ")

    def _profile_filename_matches_gcode(self, profile_path, gcode_path=None):
        profile_stem = self._strip_kc_profile_suffixes(profile_path)
        gcode_stem = self._strip_kc_profile_suffixes(gcode_path or self.gcode_nc_path_var.get())
        if not profile_stem or not gcode_stem:
            return False
        return gcode_stem in profile_stem or profile_stem in gcode_stem

    def _profile_candidate_matches_gcode(self, profile_path, gcode_path=None):
        target_path = str(profile_path or "").strip()
        if not target_path or not os.path.exists(target_path):
            return False, False
        try:
            profile = self._load_kc_profile_payload_from_path(target_path)
        except Exception:
            profile = None
        if not isinstance(profile, dict):
            return False, self._profile_filename_matches_gcode(target_path, gcode_path)
        profile_context = self._resolve_profile_template_context(profile)
        current_context = self._build_profile_template_context(
            process_path=self._get_primary_input_file_or_empty(),
            gcode_path=gcode_path,
        )
        matches_context = self._template_context_matches_gcode_only(profile_context, current_context)
        profile_gcode = str(
            profile_context.get("gcode_path") or profile.get("gcode_path") or ""
        ).strip()
        matches_name = (
            self._profile_filename_matches_gcode(target_path, gcode_path)
            or self._profile_filename_matches_gcode(profile_gcode, gcode_path)
        )
        return bool(matches_context), bool(matches_name)

    def _is_supported_kc_profile_file(self, file_name):
        name = os.path.basename(str(file_name or "").strip()).lower()
        return bool(name.endswith(".kcke") or name.endswith(".kcke.json") or name.endswith(".json"))

    def _load_kc_profile_payload_from_path(self, file_path):
        target_path = str(file_path or "").strip()
        if not target_path or not os.path.exists(target_path):
            return None
        with open(target_path, "r", encoding="utf-8") as infile:
            payload = json.load(infile)
        if not isinstance(payload, dict):
            return None
        profile = payload.get("profile")
        if isinstance(profile, dict):
            return self._normalize_loaded_kc_profile(dict(profile), source_path=target_path)
        if self._profile_has_saved_payload(payload):
            return self._normalize_loaded_kc_profile(dict(payload), source_path=target_path)
        fallback_keys = {
            "template_context",
            "global_kc",
            "global_ke",
            "ke_value",
            "global_idle",
            "point_kc_map",
            "point_actual_feed_map",
            "point_actual_mrr_map",
            "sample_forward_semantics",
            "feed_source_semantics",
            "interval_templates",
            "pit_records",
            "segment_records",
        }
        if any(key in payload for key in fallback_keys):
            return self._normalize_loaded_kc_profile(dict(payload), source_path=target_path)
        return None

    def _prune_saved_kc_profile_index(self):
        profile_index = getattr(self, "saved_kc_profile_index", None)
        if not isinstance(profile_index, dict):
            self.saved_kc_profile_index = {}
            return

        cleaned = {}
        for raw_key, raw_entry in profile_index.items():
            profile_key = str(raw_key or "").strip()
            if not profile_key or not isinstance(raw_entry, dict):
                continue
            profile_path = str(raw_entry.get("profile_path") or "").strip()
            if not profile_path or not os.path.exists(profile_path):
                continue
            cleaned[profile_key] = {
                "profile_path": profile_path,
                "process_path": str(raw_entry.get("process_path") or "").strip(),
                "gcode_path": str(raw_entry.get("gcode_path") or "").strip(),
                "updated_at": str(raw_entry.get("updated_at") or "").strip(),
                "source": str(raw_entry.get("source") or "").strip(),
            }
        self.saved_kc_profile_index = cleaned

    def _build_saved_kc_profile_index_entry(self, profile_path, profile):
        target_path = str(profile_path or "").strip()
        if not target_path:
            return None
        normalized_profile = self._normalize_loaded_kc_profile(profile, source_path=target_path)
        template_context = self._resolve_profile_template_context(normalized_profile)
        return {
            "profile_path": target_path,
            "process_path": str(template_context.get("process_path") or (normalized_profile or {}).get("process_path") or ""),
            "gcode_path": str(template_context.get("gcode_path") or (normalized_profile or {}).get("gcode_path") or ""),
            "updated_at": str((normalized_profile or {}).get("updated_at") or ""),
            "source": str((normalized_profile or {}).get("source") or ""),
        }

    def _register_saved_kc_profile_index(self, profile_key, profile_path, profile, persist=True):
        key_text = str(profile_key or "").strip()
        entry = self._build_saved_kc_profile_index_entry(profile_path, profile)
        if not key_text or not entry:
            return False
        profile_index = getattr(self, "saved_kc_profile_index", None)
        if not isinstance(profile_index, dict):
            self.saved_kc_profile_index = {}
            profile_index = self.saved_kc_profile_index
        profile_index[key_text] = entry
        if persist:
            self._persist_app_config()
        return True

    def _write_cached_kc_profile(self, profile_key, profile, process_path=None):
        cache_path = self._build_kc_profile_cache_path(profile_key, process_path=process_path)
        if not cache_path:
            return ""
        payload = self._wrap_kc_profile_file_payload(profile, active_profile_path="")
        with open(cache_path, "w", encoding="utf-8") as outfile:
            json.dump(payload, outfile, ensure_ascii=False, separators=(",", ":"))
        return cache_path

    def _load_saved_kc_profile_from_index(self, profile_key):
        key_text = str(profile_key or "").strip()
        if not key_text:
            return None
        entry = (getattr(self, "saved_kc_profile_index", {}) or {}).get(key_text)
        if not isinstance(entry, dict):
            return None
        profile_path = str(entry.get("profile_path") or "").strip()
        if not profile_path or not os.path.exists(profile_path):
            (getattr(self, "saved_kc_profile_index", {}) or {}).pop(key_text, None)
            return None
        try:
            profile = self._load_kc_profile_payload_from_path(profile_path)
        except Exception:
            return None
        if not isinstance(profile, dict):
            return None
        self.saved_kc_profiles[key_text] = profile
        return profile

    def _migrate_legacy_saved_kc_profiles(self, saved_profiles, persist=True):
        if not isinstance(saved_profiles, dict):
            return False
        migrated = False
        for raw_key, raw_profile in saved_profiles.items():
            profile_key = str(raw_key or "").strip()
            profile = self._normalize_loaded_kc_profile(dict(raw_profile), source_path="") if isinstance(raw_profile, dict) else None
            if not profile_key or not self._profile_has_saved_payload(profile):
                continue
            try:
                cache_path = self._write_cached_kc_profile(profile_key, profile, process_path=profile.get("process_path"))
            except Exception:
                continue
            if not cache_path:
                continue
            self.saved_kc_profiles[profile_key] = profile
            self._register_saved_kc_profile_index(profile_key, cache_path, profile, persist=False)
            migrated = True
        if migrated and persist:
            self._persist_app_config()
        return migrated

    def _profile_matches_current_context(self, profile, process_path=None, gcode_path=None):
        if not isinstance(profile, dict):
            return False
        profile_context = self._resolve_profile_template_context(profile)
        if not profile_context:
            return False
        current_context = self._build_profile_template_context(
            process_path=process_path,
            gcode_path=gcode_path,
        )
        current_has_gcode = bool(
            str(current_context.get("gcode_hash") or "").strip()
            or str(current_context.get("gcode_path") or "").strip()
        )
        profile_has_gcode = bool(
            str(profile_context.get("gcode_hash") or "").strip()
            or str(profile_context.get("gcode_path") or "").strip()
        )
        if current_has_gcode and profile_has_gcode:
            return self._template_context_matches_gcode_only(profile_context, current_context)
        return self._template_context_matches_process_only(profile_context, current_context)

    def _resolve_imported_profile_for_current_context(self, process_path=None, allow_autoload=True):
        imported_profile = getattr(self, "imported_kc_profile", None)
        imported_path = str(getattr(self, "imported_kc_profile_path", "") or "").strip()
        if isinstance(imported_profile, dict):
            imported_profile = self._normalize_loaded_kc_profile(imported_profile, source_path=imported_path)
            self.imported_kc_profile = imported_profile
        if (
            isinstance(imported_profile, dict)
            and self._profile_has_saved_payload(imported_profile)
            and self._profile_matches_current_context(imported_profile, process_path=process_path)
        ):
            if self._get_profile_origin() != "imported_profile":
                self._activate_profile_state(
                    imported_profile,
                    origin="imported_profile",
                    file_path=imported_path,
                )
            return imported_profile
        if not allow_autoload:
            return None
        if self._should_skip_auto_profile_for_current_gcode():
            return None
        gcode_var = getattr(self, "gcode_nc_path_var", None)
        has_gcode_accessor = hasattr(gcode_var, "get")
        gcode_value = ""
        if has_gcode_accessor:
            try:
                gcode_value = str(gcode_var.get() or "").strip()
            except Exception:
                gcode_value = ""
        if not gcode_value and not str(imported_path or "").strip():
            return None

        for candidate_path in self._collect_kc_profile_file_candidates_for_gcode():
            try:
                loaded_profile = self._load_kc_profile_payload_from_path(candidate_path)
            except Exception:
                loaded_profile = None
            if not isinstance(loaded_profile, dict):
                continue
            matches_context, matches_name = self._profile_candidate_matches_gcode(candidate_path, gcode_value)
            if not matches_context:
                if not matches_name:
                    continue
                if self._profile_has_gcode_context(loaded_profile):
                    continue
            self._activate_profile_state(
                loaded_profile,
                origin="imported_profile",
                file_path=str(candidate_path),
            )
            self._register_gcode_profile_binding(
                loaded_profile.get("gcode_path") or self.gcode_nc_path_var.get(),
                str(candidate_path),
                persist=False,
            )
            return getattr(self, "imported_kc_profile", None)
        return None

    def _resolve_runtime_identified_profile_for_current_case(self, measurement=None, process_path=None):
        runtime_profile = getattr(self, "runtime_identified_kc_profile", None)
        if not isinstance(runtime_profile, dict):
            return None
        if not self._profile_has_saved_payload(runtime_profile):
            return None
        if not self._profile_matches_current_context(runtime_profile, process_path=process_path):
            return None
        current_case_signature = self._get_current_measurement_case_signature(measurement)
        stored_case_signature = str(getattr(self, "runtime_identified_profile_case_signature", "") or "")
        if current_case_signature and stored_case_signature and current_case_signature != stored_case_signature:
            return None
        return runtime_profile

    def _resolve_forward_prediction_profile(self, measurement=None, process_path=None, allow_autoload_imported=True):
        if allow_autoload_imported is None:
            allow_autoload_imported = self._should_allow_imported_profile_autoload()
        origin = self._get_profile_origin()
        if origin == "imported_profile":
            profile = self._resolve_imported_profile_for_current_context(
                process_path=process_path,
                allow_autoload=allow_autoload_imported,
            )
            if isinstance(profile, dict):
                return "imported_profile", profile
            self._clear_active_profile_state()
        elif origin == "runtime_identified_profile":
            profile = self._resolve_runtime_identified_profile_for_current_case(
                measurement=measurement,
                process_path=process_path,
            )
            if isinstance(profile, dict):
                return "runtime_identified_profile", profile
            self._clear_runtime_identified_profile_state(clear_active=True, reason="runtime_case_mismatch")

        if allow_autoload_imported:
            profile = self._resolve_imported_profile_for_current_context(
                process_path=process_path,
                allow_autoload=True,
            )
            if isinstance(profile, dict):
                return "imported_profile", profile
        return "no_profile", None

    def _get_saved_kc_profile_for_input(self, process_path=None):
        profile = self._resolve_imported_profile_for_current_context(
            process_path=process_path,
            allow_autoload=self._should_allow_imported_profile_autoload(),
        )
        if isinstance(profile, dict):
            self._debug_prediction_state_event(
                "resolve_imported_profile",
                kc_map_source="imported_profile",
            )
        return profile if isinstance(profile, dict) else None

    def has_prediction_model_ready(self, process_path=None):
        if self.has_identified_kc_ke():
            return True
        profile = self._get_saved_kc_profile_for_input(process_path)
        return self._profile_has_saved_payload(profile)

    def _get_saved_kc_profile_signature(self, process_path=None):
        profile = self._get_saved_kc_profile_for_input(process_path)
        if not profile:
            return ""
        return self._build_profile_runtime_signature(
            profile,
            profile_path=str(
                getattr(self, "imported_kc_profile_path", "")
                or getattr(self, "active_kc_profile_path", "")
                or ""
            ),
        )

    def _normalize_profile_binding_path(self, path):
        target_path = str(path or "").strip()
        if not target_path:
            return ""
        try:
            return os.path.normcase(os.path.abspath(target_path))
        except Exception:
            return os.path.normcase(target_path)

    def _compute_profile_context_hash(self, path):
        target_path = str(path or "").strip()
        if not target_path or not os.path.exists(target_path):
            return ""
        import hashlib

        digest = hashlib.sha1()
        with open(target_path, "rb") as infile:
            while True:
                chunk = infile.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def _build_profile_template_context(self, process_path=None, gcode_path=None):
        process_target = str(process_path or self.get_primary_input_file() or "").strip()
        gcode_target = str(
            gcode_path
            if gcode_path is not None
            else (
                self.gcode_nc_path_var.get().strip()
                if hasattr(getattr(self, "gcode_nc_path_var", None), "get")
                else ""
            )
        ).strip()

        def _normalize_path_value(value):
            if not value:
                return ""
            try:
                return os.path.abspath(value)
            except Exception:
                return str(value)

        process_abs = _normalize_path_value(process_target)
        gcode_abs = _normalize_path_value(gcode_target)
        return {
            "process_path": process_abs,
            "process_hash": self._compute_profile_context_hash(process_abs),
            "gcode_path": gcode_abs,
            "gcode_hash": self._compute_profile_context_hash(gcode_abs),
        }

    def _resolve_profile_template_context(self, profile):
        if not isinstance(profile, dict):
            return {}
        raw_context = dict(profile.get("template_context") or {}) if isinstance(profile.get("template_context"), dict) else {}
        process_path = raw_context.get("process_path") or profile.get("process_path")
        gcode_path = raw_context.get("gcode_path") or profile.get("gcode_path")
        process_hash = str(raw_context.get("process_hash") or profile.get("process_hash") or "").strip().lower()
        gcode_hash = str(raw_context.get("gcode_hash") or profile.get("gcode_hash") or "").strip().lower()
        resolved = self._build_profile_template_context(process_path=process_path, gcode_path=gcode_path)
        if process_hash:
            resolved["process_hash"] = process_hash
        if gcode_hash:
            resolved["gcode_hash"] = gcode_hash
        return resolved

    def _profile_has_gcode_context(self, profile):
        if not isinstance(profile, dict):
            return False
        context = self._resolve_profile_template_context(profile)
        return bool(
            str(context.get("gcode_hash") or profile.get("gcode_hash") or "").strip()
            or str(context.get("gcode_path") or profile.get("gcode_path") or "").strip()
        )

    def _current_has_gcode_context(self, gcode_path=None):
        context = self._build_profile_template_context(
            process_path=self._get_primary_input_file_or_empty(),
            gcode_path=gcode_path,
        )
        return bool(
            str(context.get("gcode_hash") or "").strip()
            or str(context.get("gcode_path") or "").strip()
        )

    def _template_context_matches_gcode_only(self, profile_context, current_context):
        if not isinstance(profile_context, dict) or not isinstance(current_context, dict):
            return False
        profile_hash = str(profile_context.get("gcode_hash") or "").strip().lower()
        current_hash = str(current_context.get("gcode_hash") or "").strip().lower()
        if profile_hash and current_hash:
            return profile_hash == current_hash
        profile_path = self._normalize_profile_binding_path(profile_context.get("gcode_path"))
        current_path = self._normalize_profile_binding_path(current_context.get("gcode_path"))
        if profile_path and current_path:
            return profile_path == current_path
        return False

    def _template_context_matches_process_only(self, profile_context, current_context):
        if not isinstance(profile_context, dict) or not isinstance(current_context, dict):
            return False
        profile_hash = str(profile_context.get("process_hash") or "").strip().lower()
        current_hash = str(current_context.get("process_hash") or "").strip().lower()
        if profile_hash and current_hash:
            return profile_hash == current_hash
        profile_path = self._normalize_profile_binding_path(profile_context.get("process_path"))
        current_path = self._normalize_profile_binding_path(current_context.get("process_path"))
        if profile_path and current_path:
            return profile_path == current_path
        return False

    def _profile_matches_current_gcode(self, profile, gcode_path=None):
        if not isinstance(profile, dict):
            return False
        profile_context = self._resolve_profile_template_context(profile)
        current_context = self._build_profile_template_context(
            process_path=self._get_primary_input_file_or_empty(),
            gcode_path=gcode_path,
        )
        return self._template_context_matches_gcode_only(profile_context, current_context)

    def _template_context_matches(self, profile_context, current_context):
        if not isinstance(profile_context, dict) or not isinstance(current_context, dict):
            return False
        current_has_gcode = bool(
            str(current_context.get("gcode_path") or "").strip()
            or str(current_context.get("gcode_hash") or "").strip()
        )
        profile_has_gcode = bool(
            str(profile_context.get("gcode_path") or "").strip()
            or str(profile_context.get("gcode_hash") or "").strip()
        )
        if current_has_gcode and profile_has_gcode:
            return self._template_context_matches_gcode_only(profile_context, current_context)
        return self._template_context_matches_process_only(profile_context, current_context)

    def _strip_measurement_bound_fields_from_profile(self, profile, keep_sample_kc_profile=False):
        if not isinstance(profile, dict):
            return profile
        cleaned = dict(profile)
        for key in (
            "measurement_binding",
            "measurement_case_signature",
            "measurement_runtime",
            "fit_df",
            "prediction_cache",
        ):
            cleaned.pop(key, None)
        if not keep_sample_kc_profile:
            cleaned.pop("sample_kc_profile", None)
        return cleaned

    def _normalize_loaded_kc_profile(self, profile, source_path=""):
        if not isinstance(profile, dict):
            return None

        source_profile = dict(
            self._strip_measurement_bound_fields_from_profile(
                profile,
                keep_sample_kc_profile=True,
            ) or {}
        )
        has_process_data = bool(getattr(self, "data", None))
        raw_segment_records = [
            dict(record)
            for record in (source_profile.get("segment_records", []) or [])
            if isinstance(record, dict)
        ]
        raw_pit_records = [
            dict(record)
            for record in (source_profile.get("pit_records", []) or [])
            if isinstance(record, dict)
        ]
        raw_template_records = [
            dict(record)
            for record in (source_profile.get("interval_templates", []) or [])
            if isinstance(record, dict)
        ]

        if raw_segment_records:
            segment_records = self._extract_profile_segment_records(source_profile)
            steady_records = self._get_steady_interval_records(segment_records)
            pit_records = self._materialize_profile_pit_records(steady_records) if has_process_data else steady_records
            interval_templates = self._build_interval_templates_for_profile(pit_records)
        else:
            legacy_records = raw_pit_records if raw_pit_records else raw_template_records
            pit_records = self._materialize_profile_pit_records(legacy_records) if has_process_data else legacy_records
            if not pit_records:
                pit_records = legacy_records
            pit_records = self._get_steady_interval_records(pit_records)
            interval_templates = self._build_interval_templates_for_profile(pit_records)
            segment_records = self._extract_profile_segment_records(source_profile)
            if not segment_records and pit_records and has_process_data:
                segment_records = self._build_profile_segment_records(interval_records=pit_records)

        template_context = self._resolve_profile_template_context(source_profile)
        normalized = {
            key: value
            for key, value in source_profile.items()
            if key not in {
                "measurement_runtime",
                "fit_df",
                "prediction_cache",
            }
        }
        normalized["template_schema_version"] = int(source_profile.get("template_schema_version", 3) or 3)
        normalized["template_context"] = template_context
        normalized["process_path"] = str(template_context.get("process_path") or source_profile.get("process_path") or "")
        normalized["gcode_path"] = str(template_context.get("gcode_path") or source_profile.get("gcode_path") or "")
        normalized["interval_templates"] = self._serialize_record_list(interval_templates)
        normalized["pit_records"] = self._serialize_record_list(pit_records)
        normalized["segment_records"] = self._serialize_record_list(segment_records)
        normalized["interval_count"] = int(len(normalized["pit_records"]))
        normalized["segment_count"] = int(len(normalized["segment_records"]))
        point_kc_map = {}
        for raw_key, raw_value in dict(source_profile.get("point_kc_map") or {}).items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key_text = f"{int(raw_key[0])}:{int(raw_key[1])}"
            else:
                key_text = str(raw_key or "").strip()
            if not key_text:
                continue
            point_kc_map[key_text] = self._json_safe_value(raw_value)
        normalized["point_kc_map"] = point_kc_map
        normalized["point_kc_count"] = int(len(point_kc_map))
        normalized["point_actual_feed_map"] = self._serialize_numeric_map(
            source_profile.get("point_actual_feed_map", {}),
            min_value=0.0,
        )
        normalized["point_actual_mrr_map"] = self._serialize_numeric_map(
            source_profile.get("point_actual_mrr_map", {}),
            min_value=0.0,
        )
        normalized["sample_forward_semantics"] = str(source_profile.get("sample_forward_semantics") or "")
        normalized["feed_source_semantics"] = str(source_profile.get("feed_source_semantics") or "")
        if isinstance(source_profile.get("line_kc_map"), dict):
            normalized["line_kc_map"] = {
                str(key): self._json_safe_value(value)
                for key, value in dict(source_profile.get("line_kc_map") or {}).items()
            }
        if isinstance(source_profile.get("sample_kc_profile"), dict):
            normalized["sample_kc_profile"] = dict(source_profile.get("sample_kc_profile") or {})
        if source_path and not normalized.get("source"):
            normalized["source"] = f"loaded:{os.path.basename(source_path)}"
        return normalized

    def _set_profile_import_skip_state(self, gcode_path=None, skipped=True):
        gcode_value = gcode_path
        if gcode_value is None:
            gcode_var = getattr(self, "gcode_nc_path_var", None)
            if hasattr(gcode_var, "get"):
                gcode_value = gcode_var.get()
        normalized_gcode = self._normalize_profile_binding_path(gcode_value)
        self._skipped_kc_profile_gcode = normalized_gcode if skipped and normalized_gcode else ""

    def _should_skip_auto_profile_for_current_gcode(self):
        skipped_gcode = str(getattr(self, "_skipped_kc_profile_gcode", "") or "").strip()
        if not skipped_gcode:
            return False
        gcode_var = getattr(self, "gcode_nc_path_var", None)
        current_gcode = self._normalize_profile_binding_path(gcode_var.get() if hasattr(gcode_var, "get") else "")
        return bool(current_gcode and current_gcode == skipped_gcode)

    def _prune_gcode_profile_bindings(self):
        bindings = getattr(self, "gcode_profile_bindings", None)
        if not isinstance(bindings, dict):
            self.gcode_profile_bindings = {}
            return
        cleaned = {}
        for gcode_key, profile_paths in bindings.items():
            normalized_key = self._normalize_profile_binding_path(gcode_key)
            if not normalized_key:
                continue
            if isinstance(profile_paths, str):
                candidate_paths = [profile_paths]
            elif isinstance(profile_paths, list):
                candidate_paths = profile_paths
            else:
                continue
            seen = set()
            valid_paths = []
            for raw_path in candidate_paths:
                profile_path = str(raw_path or "").strip()
                if not profile_path:
                    continue
                normalized_profile = self._normalize_profile_binding_path(profile_path)
                if not normalized_profile or normalized_profile in seen:
                    continue
                if not os.path.exists(profile_path):
                    continue
                seen.add(normalized_profile)
                valid_paths.append(profile_path)
            if valid_paths:
                cleaned[normalized_key] = valid_paths
        self.gcode_profile_bindings = cleaned

    def _register_gcode_profile_binding(self, gcode_path=None, profile_path=None, persist=True):
        normalized_gcode = self._normalize_profile_binding_path(gcode_path or self.gcode_nc_path_var.get())
        profile_path = str(profile_path or getattr(self, "active_kc_profile_path", "") or "").strip()
        if not normalized_gcode or not profile_path:
            return False
        if not os.path.exists(profile_path):
            return False
        self._prune_gcode_profile_bindings()
        binding_paths = list(self.gcode_profile_bindings.get(normalized_gcode, []))
        normalized_profile = self._normalize_profile_binding_path(profile_path)
        binding_paths = [
            item for item in binding_paths
            if self._normalize_profile_binding_path(item) != normalized_profile and os.path.exists(item)
        ]
        binding_paths.insert(0, profile_path)
        self.gcode_profile_bindings[normalized_gcode] = binding_paths
        if persist:
            self._persist_app_config()
        return True

    def _collect_kc_profile_file_candidates_for_gcode(self, gcode_path=None):
        normalized_gcode = self._normalize_profile_binding_path(gcode_path or self.gcode_nc_path_var.get())
        if not normalized_gcode:
            return []

        self._prune_gcode_profile_bindings()
        candidate_meta = []
        seen = set()

        def _append_candidate(profile_path, rank_bias=100):
            target_path = str(profile_path or "").strip()
            if not target_path or not os.path.exists(target_path):
                return
            normalized_profile = self._normalize_profile_binding_path(target_path)
            if not normalized_profile or normalized_profile in seen:
                return
            matches_context, matches_name = self._profile_candidate_matches_gcode(target_path, gcode_path)
            if not matches_context and not matches_name:
                return
            seen.add(normalized_profile)
            rank = int(rank_bias)
            if matches_context:
                rank -= 20
            if matches_name:
                rank -= 8
            if normalized_profile == self._normalize_profile_binding_path(getattr(self, "active_kc_profile_path", "")):
                rank -= 4
            candidate_meta.append((rank, len(candidate_meta), target_path))

        for profile_path in self.gcode_profile_bindings.get(normalized_gcode, []):
            _append_candidate(profile_path, rank_bias=0)

        active_profile_path = str(getattr(self, "active_kc_profile_path", "") or "").strip()
        if active_profile_path:
            _append_candidate(active_profile_path, rank_bias=4)

        search_roots = []

        def _add_search_root(base_dir, recursive):
            root_dir = str(base_dir or "").strip()
            if not root_dir or not os.path.isdir(root_dir):
                return
            normalized_root = self._normalize_profile_binding_path(root_dir)
            if not normalized_root:
                return
            for existing_root, _existing_recursive in search_roots:
                if existing_root == normalized_root:
                    return
            search_roots.append((normalized_root, bool(recursive)))

        _add_search_root(getattr(self, "kc_profile_dir", ""), True)
        if gcode_path:
            _add_search_root(os.path.dirname(os.path.abspath(gcode_path)), False)
        if active_profile_path:
            _add_search_root(os.path.dirname(os.path.abspath(active_profile_path)), False)

        for normalized_root, recursive in search_roots:
            if recursive:
                for root_dir, _dirs, file_names in os.walk(normalized_root):
                    for entry in sorted(file_names):
                        if not self._is_supported_kc_profile_file(entry):
                            continue
                        _append_candidate(os.path.join(root_dir, entry), rank_bias=40)
            else:
                for entry in sorted(os.listdir(normalized_root)):
                    if not self._is_supported_kc_profile_file(entry):
                        continue
                    _append_candidate(os.path.join(normalized_root, entry), rank_bias=60)

        candidate_meta.sort(key=lambda item: (item[0], item[1], item[2].lower()))
        return [profile_path for _rank, _order, profile_path in candidate_meta]

    def _active_kc_profile_matches_gcode(self, gcode_path=None):
        active_profile = getattr(self, "imported_kc_profile", None)
        if not self._profile_has_saved_payload(active_profile):
            return False
        if self._current_has_gcode_context(gcode_path=gcode_path):
            if self._profile_has_gcode_context(active_profile):
                return self._profile_matches_current_gcode(active_profile, gcode_path=gcode_path)
            profile_context = self._resolve_profile_template_context(active_profile)
            current_context = self._build_profile_template_context(
                process_path=self._get_primary_input_file_or_empty(),
                gcode_path=gcode_path,
            )
            return self._template_context_matches_process_only(profile_context, current_context)
        return self._profile_matches_current_context(active_profile, gcode_path=gcode_path)

    def _is_imported_profile_forward_lock_active(self):
        if self._normalize_profile_origin(getattr(self, "profile_origin", "no_profile")) != "imported_profile":
            return False
        profile = getattr(self, "imported_kc_profile", None)
        if not isinstance(profile, dict):
            profile = getattr(self, "active_kc_profile", None)
        if not self._profile_has_saved_payload(profile):
            return False
        if self._current_has_gcode_context():
            if self._profile_has_gcode_context(profile):
                return self._profile_matches_current_gcode(profile)
            return self._profile_matches_current_context(
                profile,
                process_path=self._get_primary_input_file_or_empty(),
            )
        return True

    def _json_safe_value(self, value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return value if np.isfinite(value) else None
        if isinstance(value, (int, str, bool)) or value is None:
            return value
        return value

    def _serialize_record_list(self, records):
        serialized = []
        for record in records or []:
            if not isinstance(record, dict):
                continue
            item = {}
            for key, value in record.items():
                item[str(key)] = self._json_safe_value(value)
            start_label = str(item.get("start_label") or "").strip()
            end_label = str(item.get("end_label") or "").strip()
            if not start_label and not end_label:
                start_label = str(item.get("process_start_label") or "").strip()
                end_label = str(item.get("process_end_label") or "").strip()
            if start_label or end_label:
                item["interval_range"] = f"{start_label}-{end_label}" if start_label and end_label else (start_label or end_label)
            serialized.append(item)
        return serialized

    def _serialize_numeric_map(self, raw_map, min_value=None):
        serialized = {}
        if not isinstance(raw_map, dict):
            return serialized
        lower_bound = None if min_value is None else float(min_value)
        for raw_key, raw_value in raw_map.items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key_text = f"{int(raw_key[0])}:{int(raw_key[1])}"
            else:
                key_text = str(raw_key or "").strip()
            if not key_text:
                continue
            try:
                numeric_value = float(raw_value)
            except Exception:
                continue
            if not np.isfinite(numeric_value):
                continue
            if lower_bound is not None and numeric_value < lower_bound:
                continue
            serialized[key_text] = float(numeric_value)
        return serialized

    def _serialize_numeric_map(self, raw_map, min_value=None):
        serialized = {}
        if not isinstance(raw_map, dict):
            return serialized
        lower_bound = None if min_value is None else float(min_value)
        for raw_key, raw_value in raw_map.items():
            if isinstance(raw_key, tuple) and len(raw_key) == 2:
                key_text = f"{int(raw_key[0])}:{int(raw_key[1])}"
            else:
                key_text = str(raw_key or "").strip()
            if not key_text:
                continue
            try:
                numeric_value = float(raw_value)
            except Exception:
                continue
            if not np.isfinite(numeric_value):
                continue
            if lower_bound is not None and numeric_value < lower_bound:
                continue
            serialized[key_text] = float(numeric_value)
        return serialized

    def _build_profile_point_numeric_map_from_sample_df(
        self,
        sample_df,
        value_column,
        *,
        min_value=0.0,
        positive_only=False,
    ):
        if sample_df is None or getattr(sample_df, "empty", True):
            return {}
        required_columns = {"line_no_aligned", "process_point_index", value_column}
        if not required_columns.issubset(set(sample_df.columns)):
            return {}

        frame = pd.DataFrame(
            {
                "line_no_aligned": pd.to_numeric(sample_df.get("line_no_aligned"), errors="coerce"),
                "process_point_index": pd.to_numeric(sample_df.get("process_point_index"), errors="coerce"),
                "value": pd.to_numeric(sample_df.get(value_column), errors="coerce"),
            }
        )
        valid_mask = (
            np.isfinite(frame["line_no_aligned"].to_numpy(dtype=float))
            & np.isfinite(frame["process_point_index"].to_numpy(dtype=float))
            & np.isfinite(frame["value"].to_numpy(dtype=float))
        )
        if positive_only:
            valid_mask &= frame["value"].to_numpy(dtype=float) > float(min_value)
        else:
            valid_mask &= frame["value"].to_numpy(dtype=float) >= float(min_value)
        frame = frame.loc[valid_mask].copy()
        if frame.empty:
            return {}

        grouped = (
            frame.groupby(["line_no_aligned", "process_point_index"], dropna=True)["value"]
            .median()
            .reset_index()
        )
        result = {}
        for _, row in grouped.iterrows():
            try:
                line_no = int(row["line_no_aligned"])
                point_idx = int(row["process_point_index"])
                value = float(row["value"])
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            if positive_only and value <= float(min_value):
                continue
            if not positive_only and value < float(min_value):
                continue
            result[f"{line_no}:{point_idx}"] = float(value)
        return result

    def _build_profile_point_actual_feed_map_from_current_measurement(self, sample_df):
        return self._build_profile_point_numeric_map_from_sample_df(
            sample_df,
            "feed_speed",
            min_value=0.0,
            positive_only=True,
        )

    def _build_profile_point_actual_mrr_map_from_current_measurement(self, sample_df):
        return self._build_profile_point_numeric_map_from_sample_df(
            sample_df,
            "mrr",
            min_value=0.0,
            positive_only=False,
        )

    def _parse_line_point_label(self, label):
        text = str(label or "").strip()
        if not text:
            return None, None
        match = re.match(r"^\s*(-?\d+)\s*\.\s*(\d+)\s*$", text)
        if not match:
            return None, None
        try:
            line_no = int(match.group(1))
            point_no = int(match.group(2))
        except Exception:
            return None, None
        if point_no <= 0:
            return line_no, None
        return line_no, point_no

    def _get_current_sample_line_point_context(self, line_numbers=None):
        context_cache_key = None
        if line_numbers is None:
            sample_line_source = getattr(self, "sample_data_line_numbers", None)
            sample_lines = np.asarray(sample_line_source if sample_line_source is not None else [], dtype=int)
            blocks = getattr(self, "sample_data_base_blocks", None)
            use_global_context = True
            context_cache_key = (
                id(sample_line_source),
                id(blocks),
                id(getattr(self, "sample_data_point_indices", None)),
                id(getattr(self, "sample_data_x_positions", None)),
                id(getattr(self, "sample_data_time_indices", None)),
            )
            cached_context = getattr(self, "_sample_line_point_context_cache", None)
            if isinstance(cached_context, dict) and cached_context.get("key") == context_cache_key:
                cached_value = cached_context.get("value")
                if cached_value:
                    return cached_value
        else:
            sample_lines = np.asarray(line_numbers, dtype=int)
            global_lines = np.asarray(getattr(self, "sample_data_line_numbers", []), dtype=int)
            use_global_context = global_lines.size == sample_lines.size and np.array_equal(global_lines, sample_lines)
            blocks = getattr(self, "sample_data_base_blocks", None) if use_global_context else None

        if sample_lines.size == 0:
            return None

        point_indices = getattr(self, "sample_data_point_indices", None)
        if use_global_context and point_indices is not None and len(point_indices) == len(sample_lines):
            sample_points = np.asarray(point_indices, dtype=int)
        else:
            sample_points = np.asarray(self.compute_line_point_indices(sample_lines, blocks=blocks), dtype=int)

        sample_x_positions = getattr(self, "sample_data_x_positions", None)
        if use_global_context and sample_x_positions is not None and len(sample_x_positions) == len(sample_lines):
            sample_x_positions = np.asarray(sample_x_positions, dtype=float)
        else:
            sample_x_positions = None
            if hasattr(self, "compute_line_x_positions"):
                try:
                    local_blocks = self.compute_sequence_blocks(sample_lines) if hasattr(self, "compute_sequence_blocks") else None
                    sample_x_positions = np.asarray(
                        self.compute_line_x_positions(sample_lines, blocks=local_blocks),
                        dtype=float,
                    )
                except Exception:
                    sample_x_positions = None

        sample_time_positions = self.get_sample_time_indices_array()
        if use_global_context and sample_time_positions is not None and len(sample_time_positions) == len(sample_lines):
            sample_time_positions = np.asarray(sample_time_positions, dtype=float)
        else:
            sample_time_positions = np.arange(len(sample_lines), dtype=float)

        point_widths = getattr(self, "sample_data_point_widths", None)
        if use_global_context and point_widths is not None and len(point_widths) == len(sample_lines):
            point_widths = np.asarray(point_widths, dtype=float)
        else:
            point_widths = np.asarray(self.compute_line_point_widths(sample_lines), dtype=float)
            if use_global_context:
                self.sample_data_point_widths = point_widths

        context = {
            "line_numbers": sample_lines,
            "point_indices": sample_points,
            "point_numbers": sample_points + 1,
            "x_positions": sample_x_positions,
            "time_positions": sample_time_positions,
            "point_widths": point_widths,
        }
        # 样本通常远多于工艺点。按程序行预建索引后，区间端点投影只需
        # 检查目标行内的少量样本，避免每个区间反复扫描整份实测数据。
        sort_order = np.argsort(sample_lines, kind="stable")
        sorted_lines = sample_lines[sort_order]
        unique_lines, first_positions = np.unique(sorted_lines, return_index=True)
        end_positions = np.append(first_positions[1:], sorted_lines.size)
        context["line_index_lookup"] = {
            int(line_no): sort_order[int(start_pos):int(end_pos)]
            for line_no, start_pos, end_pos in zip(
                unique_lines,
                first_positions,
                end_positions,
            )
        }
        if context_cache_key is not None:
            self._sample_line_point_context_cache = {
                "key": context_cache_key,
                "value": context,
            }
        return context

    @staticmethod
    def _get_process_row_sample_line(row, fallback=0):
        """返回与实际负载文件一致的程序行号口径。"""
        value = row.get("line_no_raw") if isinstance(row, dict) else None
        if value is None and isinstance(row, dict):
            value = row.get("line_no_aligned")
        try:
            return int(value)
        except Exception:
            return int(fallback)

    def _get_process_point_anchor_x(self, line_no, point_no, process_rows=None):
        try:
            line_value = int(line_no)
        except Exception:
            return float("nan")
        if point_no is None:
            return float(line_value)

        source_rows = process_rows if process_rows is not None else (self.data or [])
        rows = source_rows if isinstance(source_rows, list) else list(source_rows)
        if not rows:
            return float("nan")
        if process_rows is None and hasattr(self, "_ensure_process_point_metadata"):
            try:
                self._ensure_process_point_metadata()
                rows = self.data or []
            except Exception:
                rows = self.data or []

        target_point_idx = max(int(point_no) - 1, 0)
        if process_rows is None or rows is getattr(self, "data", None):
            try:
                bucket = self._build_process_point_lookup().get(line_value)
            except Exception:
                bucket = None
            if isinstance(bucket, dict):
                point_indices = np.asarray(bucket.get("process_point_index", []), dtype=int)
                anchor_values = np.asarray(bucket.get("process_anchor_x", []), dtype=float)
                exact = np.flatnonzero(point_indices == target_point_idx)
                if exact.size and anchor_values.size == point_indices.size:
                    return float(anchor_values[int(exact[0])])
                point_count = int(bucket.get("point_count", point_indices.size) or point_indices.size)
                if point_count > 0:
                    safe_point_idx = max(0, min(target_point_idx, point_count - 1))
                    return float(line_value) + float(safe_point_idx) / float(point_count)

        same_line_rows = []
        for row_idx, row in enumerate(rows):
            row_line = self._get_process_row_sample_line(row, fallback=row_idx)
            if row_line != line_value:
                continue
            try:
                point_idx = int(row.get("process_point_index", len(same_line_rows)))
            except Exception:
                point_idx = int(len(same_line_rows))
            try:
                point_count = int(row.get("process_point_count", 0) or 0)
            except Exception:
                point_count = 0
            same_line_rows.append((point_idx, point_count))
            if point_idx == target_point_idx and point_count > 0:
                return float(line_value) + float(point_idx) / float(point_count)

        if not same_line_rows:
            return float("nan")

        same_line_rows.sort(key=lambda item: item[0])
        total_points = max(
            int(max((item[1] for item in same_line_rows), default=0) or 0),
            int(same_line_rows[-1][0]) + 1,
        )
        if total_points <= 0:
            return float(line_value)
        target_point_idx = max(0, min(target_point_idx, total_points - 1))
        return float(line_value) + float(target_point_idx) / float(total_points)

    def _get_sample_point_anchor_x(self, line_no, point_no, sample_line_numbers=None):
        try:
            line_value = int(line_no)
        except Exception:
            return float("nan")
        if point_no is None:
            return float(line_value)

        context = self._get_current_sample_line_point_context(line_numbers=sample_line_numbers)
        if not context:
            return float("nan")
        point_numbers = np.asarray(context.get("point_numbers", []), dtype=int)
        line_numbers = np.asarray(context.get("line_numbers", []), dtype=int)
        if point_numbers.size == 0 or point_numbers.size != line_numbers.size:
            return float("nan")

        point_value = max(int(point_no), 1)
        exact_match = np.flatnonzero((line_numbers == line_value) & (point_numbers == point_value))
        x_positions = np.asarray(context.get("x_positions", []), dtype=float)
        if exact_match.size and x_positions.size == line_numbers.size:
            return float(x_positions[int(exact_match[0])])

        same_line = np.flatnonzero(line_numbers == line_value)
        if same_line.size == 0:
            return float("nan")
        sample_count = int(same_line.size)
        point_value = max(1, min(point_value, sample_count))
        return float(line_value) + float(point_value - 1) / float(sample_count)

    def _resolve_interval_process_x_bounds(self, interval, process_bounds=None, process_rows=None):
        resolved_bounds = process_bounds if isinstance(process_bounds, dict) else self._resolve_interval_process_bounds(interval, process_rows=process_rows)
        if not resolved_bounds:
            return None

        source_rows = process_rows if process_rows is not None else (self.data or [])
        rows = source_rows if isinstance(source_rows, list) else list(source_rows)
        if not rows:
            return None

        try:
            start_idx = int(resolved_bounds.get("start_idx"))
            end_idx = int(resolved_bounds.get("end_idx"))
            start_line = int(resolved_bounds.get("start_line"))
            end_line = int(resolved_bounds.get("end_line"))
            start_point_idx = int(resolved_bounds.get("start_point_index", 0))
            end_point_idx = int(resolved_bounds.get("end_point_index", 0))
        except Exception:
            return None

        def _anchor_from_row(row_idx, line_no, point_idx):
            try:
                point_count = int(rows[int(row_idx)].get("process_point_count", 0) or 0)
            except Exception:
                point_count = 0
            if point_count > 0:
                return float(line_no) + float(max(int(point_idx), 0)) / float(point_count)
            return self._get_process_point_anchor_x(
                line_no,
                int(point_idx) + 1,
                process_rows=rows,
            )

        process_start_x = _anchor_from_row(start_idx, start_line, start_point_idx)
        process_end_x = _anchor_from_row(end_idx, end_line, end_point_idx)

        process_display_end_x = float("nan")
        if 0 <= end_idx + 1 < len(rows):
            next_row = rows[end_idx + 1]
            next_line = self._get_process_row_sample_line(next_row, fallback=end_line)
            try:
                next_point_idx = int(next_row.get("process_point_index", end_point_idx + 1))
            except Exception:
                next_point_idx = end_point_idx + 1
            process_display_end_x = _anchor_from_row(end_idx + 1, next_line, next_point_idx)
        if not np.isfinite(process_display_end_x):
            try:
                point_count = int(rows[end_idx].get("process_point_count", 0) or 0)
            except Exception:
                point_count = 0
            if point_count > 0:
                process_display_end_x = float(end_line) + float(end_point_idx + 1) / float(point_count)
        if not np.isfinite(process_display_end_x):
            process_display_end_x = float(process_end_x)
        if np.isfinite(process_end_x) and process_display_end_x <= process_end_x:
            process_display_end_x = float(process_end_x) + 1e-9

        return {
            "process_start_x": float(process_start_x) if np.isfinite(process_start_x) else float("nan"),
            "process_end_x": float(process_end_x) if np.isfinite(process_end_x) else float("nan"),
            "process_display_end_x": float(process_display_end_x) if np.isfinite(process_display_end_x) else float("nan"),
        }

    def _project_process_point_to_sample_anchor(
        self,
        *,
        process_label=None,
        process_line=None,
        process_point_no=None,
        context=None,
        process_rows=None,
        prefer="start",
        lower_x=None,
        upper_x=None,
    ):
        if process_line is None or process_point_no is None:
            parsed_line, parsed_point = self._parse_line_point_label(process_label)
            if process_line is None:
                process_line = parsed_line
            if process_point_no is None:
                process_point_no = parsed_point
        if process_line is None:
            return None

        sample_context = context if isinstance(context, dict) else self._get_current_sample_line_point_context()
        if not sample_context:
            return None

        sample_lines = np.asarray(sample_context.get("line_numbers", []), dtype=int)
        sample_points = np.asarray(sample_context.get("point_indices", []), dtype=int)
        sample_point_numbers = np.asarray(sample_context.get("point_numbers", []), dtype=int)
        sample_x = np.asarray(sample_context.get("x_positions", []), dtype=float)
        if sample_lines.size == 0 or sample_points.size != sample_lines.size or sample_x.size != sample_lines.size:
            return None

        process_x = self._get_process_point_anchor_x(process_line, process_point_no, process_rows=process_rows)
        if not np.isfinite(process_x):
            return None

        line_index_lookup = sample_context.get("line_index_lookup")
        if isinstance(line_index_lookup, dict):
            candidate_indices = np.asarray(
                line_index_lookup.get(int(process_line), []),
                dtype=int,
            )
        else:
            candidate_indices = np.flatnonzero(sample_lines == int(process_line))
        if candidate_indices.size:
            candidate_x = sample_x[candidate_indices]
            candidate_mask = np.isfinite(candidate_x)
            if np.isfinite(lower_x):
                candidate_mask &= candidate_x >= float(lower_x) - 1e-9
            if np.isfinite(upper_x):
                candidate_mask &= candidate_x <= float(upper_x) + 1e-9
            candidate_indices = candidate_indices[candidate_mask]
        if candidate_indices.size == 0:
            return None

        distances = np.abs(sample_x[candidate_indices] - float(process_x))
        best_distance = np.min(distances)
        nearest = candidate_indices[np.flatnonzero(distances == best_distance)]
        selected_idx = int(nearest[0] if str(prefer or "").strip().lower() != "end" else nearest[-1])
        return {
            "sample_idx": int(selected_idx),
            "sample_line": int(sample_lines[selected_idx]),
            "sample_point_index": int(sample_points[selected_idx]),
            "sample_point_no": int(sample_point_numbers[selected_idx]),
            "sample_label": self.format_line_point(sample_lines[selected_idx], sample_points[selected_idx]),
            "sample_x": float(sample_x[selected_idx]),
        }

    def _project_process_interval_to_sample_bounds_by_x(self, interval, context=None, process_rows=None):
        sample_context = context if isinstance(context, dict) else self._get_current_sample_line_point_context()
        if not sample_context:
            return None

        sample_lines = np.asarray(sample_context.get("line_numbers", []), dtype=int)
        sample_points = np.asarray(sample_context.get("point_indices", []), dtype=int)
        sample_x = np.asarray(sample_context.get("x_positions", []), dtype=float)
        time_positions = np.asarray(sample_context.get("time_positions", []), dtype=float)
        point_widths = np.asarray(sample_context.get("point_widths", []), dtype=float)
        if sample_lines.size == 0 or sample_points.size != sample_lines.size or sample_x.size != sample_lines.size:
            return None

        process_bounds = self._resolve_interval_process_bounds(interval, process_rows=process_rows)
        if not process_bounds:
            return None
        process_x_bounds = self._resolve_interval_process_x_bounds(interval, process_bounds=process_bounds, process_rows=process_rows)
        if not process_x_bounds:
            return None

        start_x = float(process_x_bounds.get("process_start_x"))
        end_x = float(process_x_bounds.get("process_end_x"))
        if not np.isfinite(start_x) or not np.isfinite(end_x):
            return None

        anchor_start = self._project_process_point_to_sample_anchor(
            process_label=process_bounds.get("process_start_label"),
            context=sample_context,
            process_rows=process_rows,
            prefer="start",
            lower_x=start_x,
            upper_x=end_x,
        )
        anchor_end = self._project_process_point_to_sample_anchor(
            process_label=process_bounds.get("process_end_label"),
            context=sample_context,
            process_rows=process_rows,
            prefer="end",
            lower_x=start_x,
            upper_x=end_x,
        )
        if not isinstance(anchor_start, dict) or not isinstance(anchor_end, dict):
            return None

        sample_start_idx = int(anchor_start["sample_idx"])
        sample_end_idx = int(anchor_end["sample_idx"])
        if sample_end_idx < sample_start_idx:
            sample_start_idx, sample_end_idx = sample_end_idx, sample_start_idx
            anchor_start, anchor_end = anchor_end, anchor_start

        sample_start_label = str(anchor_start.get("sample_label") or self.format_line_point(sample_lines[sample_start_idx], sample_points[sample_start_idx]))
        sample_end_label = str(anchor_end.get("sample_label") or self.format_line_point(sample_lines[sample_end_idx], sample_points[sample_end_idx]))
        display_start_x = float(sample_x[sample_start_idx])
        point_width = float(point_widths[sample_end_idx]) if point_widths.size == sample_lines.size else float("nan")
        display_end_x = float(sample_x[sample_end_idx] + max(point_width, 1e-9)) if np.isfinite(point_width) else float(end_x)
        display_start_t = float(time_positions[sample_start_idx]) if time_positions.size == sample_lines.size else float("nan")
        display_end_t = (
            float(time_positions[sample_end_idx] + 1.0)
            if time_positions.size == sample_lines.size else float("nan")
        )

        return {
            "sample_start_idx": int(sample_start_idx),
            "sample_end_idx": int(sample_end_idx),
            "sample_start_line": int(sample_lines[sample_start_idx]),
            "sample_end_line": int(sample_lines[sample_end_idx]),
            "sample_start_label": str(sample_start_label),
            "sample_end_label": str(sample_end_label),
            "sample_anchor_start_idx": int(anchor_start["sample_idx"]),
            "sample_anchor_end_idx": int(anchor_end["sample_idx"]),
            "sample_anchor_start_label": str(anchor_start.get("sample_label") or sample_start_label),
            "sample_anchor_end_label": str(anchor_end.get("sample_label") or sample_end_label),
            "start_label": str(sample_start_label),
            "end_label": str(sample_end_label),
            "display_start_x": float(display_start_x),
            "display_end_x": float(display_end_x),
            "display_start_t": float(display_start_t),
            "display_end_t": float(display_end_t),
            "sample_count": int(sample_end_idx - sample_start_idx + 1),
        }

    def _find_sample_index_for_line_point(self, line_numbers, point_numbers, line_no, point_no, prefer="start"):
        if line_numbers is None or point_numbers is None:
            return None
        if len(line_numbers) == 0 or len(point_numbers) != len(line_numbers):
            return None

        exact_match = np.flatnonzero((line_numbers == int(line_no)) & (point_numbers == int(point_no)))
        if exact_match.size:
            return int(exact_match[0] if prefer != "end" else exact_match[-1])

        same_line = np.flatnonzero(line_numbers == int(line_no))
        if same_line.size:
            local_points = point_numbers[same_line]
            nearest_idx = int(np.argmin(np.abs(local_points - int(point_no))))
            return int(same_line[nearest_idx])

        line_deltas = np.abs(line_numbers.astype(float) - float(line_no))
        finite_mask = np.isfinite(line_deltas)
        if not np.any(finite_mask):
            return None
        candidate_indices = np.flatnonzero(line_deltas == np.min(line_deltas[finite_mask]))
        if candidate_indices.size == 0:
            return None
        return int(candidate_indices[0] if prefer != "end" else candidate_indices[-1])

    def _resolve_interval_sample_bounds(self, interval, line_numbers=None):
        if self._has_authoritative_segmentation_state():
            interval_id = str(
                interval.get("interval_id") or interval.get("zone_id") or ""
            ).strip()
            current_records = getattr(self, "current_interval_records", None) or []
            sample_lines_source = getattr(self, "sample_data_line_numbers", None)
            cache_key = (
                id(current_records),
                len(current_records),
                id(sample_lines_source),
                len(sample_lines_source) if sample_lines_source is not None else 0,
                id(getattr(self, "_latest_segmentation_result", None)),
            )
            cached = getattr(
                self,
                "_authoritative_segmentation_sample_lookup_cache",
                None,
            )
            if not isinstance(cached, dict) or cached.get("key") != cache_key:
                try:
                    projected_records = self._get_authoritative_segmentation_sample_records()
                except Exception:
                    return None
                projected_lookup = {}
                for projected in projected_records:
                    projected_id = str(
                        projected.get("interval_id") or projected.get("zone_id") or ""
                    ).strip()
                    if projected_id and projected_id not in projected_lookup:
                        projected_lookup[projected_id] = dict(projected)
                cached = {"key": cache_key, "value": projected_lookup}
                self._authoritative_segmentation_sample_lookup_cache = cached
            projected_lookup = cached.get("value") or {}
            projected = projected_lookup.get(interval_id)
            return dict(projected) if isinstance(projected, dict) else None

        context = self._get_current_sample_line_point_context(line_numbers=line_numbers)
        if not context:
            return None

        sample_lines = context["line_numbers"]
        sample_points = context["point_indices"]
        point_numbers = context["point_numbers"]
        x_positions = context["x_positions"]
        time_positions = context["time_positions"]
        point_widths = context["point_widths"]
        sample_size = int(len(sample_lines))

        projected_bounds = self._project_process_interval_to_sample_bounds_by_x(interval, context=context)
        if projected_bounds:
            return projected_bounds

        start_idx = None
        end_idx = None

        start_line_from_label, start_point_no = self._parse_line_point_label(interval.get("start_label"))
        end_line_from_label, end_point_no = self._parse_line_point_label(interval.get("end_label"))
        try:
            explicit_start_idx = int(interval.get("sample_start_idx"))
            explicit_end_idx = int(interval.get("sample_end_idx"))
        except Exception:
            explicit_start_idx = None
            explicit_end_idx = None
        if (
            explicit_start_idx is not None
            and explicit_end_idx is not None
            and 0 <= explicit_start_idx < sample_size
            and 0 <= explicit_end_idx < sample_size
        ):
            explicit_start_line = int(sample_lines[explicit_start_idx])
            explicit_end_line = int(sample_lines[explicit_end_idx])
            explicit_matches = True
            if start_line_from_label is not None and explicit_start_line != int(start_line_from_label):
                explicit_matches = False
            if end_line_from_label is not None and explicit_end_line != int(end_line_from_label):
                explicit_matches = False
            if explicit_matches:
                start_idx = int(explicit_start_idx)
                end_idx = int(explicit_end_idx)
        if start_line_from_label is not None and start_point_no is not None:
            if start_idx is None:
                start_idx = self._find_sample_index_for_line_point(
                    sample_lines, point_numbers, start_line_from_label, start_point_no, prefer="start"
                )
        if end_line_from_label is not None and end_point_no is not None:
            if end_idx is None:
                end_idx = self._find_sample_index_for_line_point(
                    sample_lines, point_numbers, end_line_from_label, end_point_no, prefer="end"
                )

        if start_idx is None or end_idx is None:
            start_line = start_line_from_label if start_line_from_label is not None else interval.get("start_line")
            end_line = end_line_from_label if end_line_from_label is not None else interval.get("end_line")
            try:
                start_line = int(start_line)
                end_line = int(end_line)
            except Exception:
                start_line = None
                end_line = None
            if start_line is not None and end_line is not None:
                line_mask = (sample_lines >= min(start_line, end_line)) & (sample_lines <= max(start_line, end_line))
                matching_indices = np.flatnonzero(line_mask)
                if matching_indices.size:
                    if start_idx is None:
                        start_idx = int(matching_indices[0])
                    if end_idx is None:
                        end_idx = int(matching_indices[-1])

        if start_idx is None or end_idx is None:
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        display_start_x = float("nan")
        display_end_x = float("nan")
        if x_positions is not None and len(x_positions) == len(sample_lines):
            display_start_x = float(x_positions[start_idx])
            point_width = float(point_widths[end_idx]) if end_idx < len(point_widths) else 1.0
            display_end_x = float(x_positions[end_idx] + max(point_width, 1e-9))

        display_start_t = float("nan")
        display_end_t = float("nan")
        if time_positions is not None and len(time_positions) == len(sample_lines):
            display_start_t = float(time_positions[start_idx])
            display_end_t = float(time_positions[end_idx] + 1.0)

        return {
            "sample_start_idx": int(start_idx),
            "sample_end_idx": int(end_idx),
            "sample_start_line": int(sample_lines[start_idx]),
            "sample_end_line": int(sample_lines[end_idx]),
            "sample_start_label": self.format_line_point(sample_lines[start_idx], sample_points[start_idx]),
            "sample_end_label": self.format_line_point(sample_lines[end_idx], sample_points[end_idx]),
            "sample_anchor_start_idx": int(start_idx),
            "sample_anchor_end_idx": int(end_idx),
            "sample_anchor_start_label": self.format_line_point(sample_lines[start_idx], sample_points[start_idx]),
            "sample_anchor_end_label": self.format_line_point(sample_lines[end_idx], sample_points[end_idx]),
            "start_label": self.format_line_point(sample_lines[start_idx], sample_points[start_idx]),
            "end_label": self.format_line_point(sample_lines[end_idx], sample_points[end_idx]),
            "display_start_x": float(display_start_x),
            "display_end_x": float(display_end_x),
            "display_start_t": float(display_start_t),
            "display_end_t": float(display_end_t),
            "sample_count": int(end_idx - start_idx + 1),
        }

    def _build_interval_sample_mask(self, interval, sample_size, line_numbers=None):
        if int(sample_size) <= 0:
            return np.zeros(0, dtype=bool)
        mask = np.zeros(int(sample_size), dtype=bool)
        try:
            start_idx = int(interval.get("sample_start_idx"))
            end_idx = int(interval.get("sample_end_idx"))
        except Exception:
            start_idx = None
            end_idx = None
        if start_idx is not None and end_idx is not None:
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            if 0 <= start_idx <= end_idx < int(sample_size):
                mask[start_idx:end_idx + 1] = True
                return mask
        bounds = self._resolve_interval_sample_bounds(interval, line_numbers=line_numbers)
        if not bounds:
            return mask
        start_idx = int(bounds["sample_start_idx"])
        end_idx = int(bounds["sample_end_idx"])
        if start_idx < 0 or end_idx >= int(sample_size) or end_idx < start_idx:
            return mask
        mask[start_idx:end_idx + 1] = True
        return mask

    def _resolve_interval_process_bounds(self, interval, process_rows=None):
        # process 边界只能来自 process 侧字段/行号，禁止再用 sample 标签反推 process row。
        source_rows = process_rows if process_rows is not None else (getattr(self, "data", None) or [])
        rows = source_rows if isinstance(source_rows, list) else list(source_rows)
        if not rows:
            return None

        if process_rows is None and hasattr(self, "_ensure_process_point_metadata"):
            try:
                self._ensure_process_point_metadata()
                rows = getattr(self, "data", None) or []
            except Exception:
                rows = getattr(self, "data", None) or []

        process_line_arr = None
        process_point_arr = None

        def _ensure_process_arrays():
            nonlocal process_line_arr, process_point_arr
            if process_line_arr is not None and process_point_arr is not None:
                return True
            process_lines = []
            process_points = []
            for row_idx, row in enumerate(rows):
                line_no = self._get_process_row_sample_line(row, fallback=row_idx)
                try:
                    point_idx = int(row.get("process_point_index", 0))
                except Exception:
                    point_idx = 0
                process_lines.append(int(line_no))
                process_points.append(max(int(point_idx), 0))
            process_line_arr = np.asarray(process_lines, dtype=int)
            process_point_arr = np.asarray(process_points, dtype=int)
            return bool(
                process_line_arr.size > 0
                and process_point_arr.size == process_line_arr.size
            )

        def _pick_start(line_no, point_no=None):
            if line_no is None or not _ensure_process_arrays():
                return None
            candidates = np.flatnonzero(process_line_arr == int(line_no))
            if candidates.size == 0:
                return None
            if point_no is None:
                return int(candidates[0])
            point_idx = max(int(point_no) - 1, 0)
            exact = candidates[process_point_arr[candidates] >= point_idx]
            if exact.size:
                return int(exact[0])
            nearest = int(np.argmin(np.abs(process_point_arr[candidates] - point_idx)))
            return int(candidates[nearest])

        def _pick_end(line_no, point_no=None):
            if line_no is None or not _ensure_process_arrays():
                return None
            candidates = np.flatnonzero(process_line_arr == int(line_no))
            if candidates.size == 0:
                return None
            if point_no is None:
                return int(candidates[-1])
            point_idx = max(int(point_no) - 1, 0)
            exact = candidates[process_point_arr[candidates] <= point_idx]
            if exact.size:
                return int(exact[-1])
            nearest = int(np.argmin(np.abs(process_point_arr[candidates] - point_idx)))
            return int(candidates[nearest])

        start_idx = None
        end_idx = None

        try:
            explicit_start_idx = int(interval.get("start_idx"))
            explicit_end_idx = int(interval.get("end_idx"))
        except Exception:
            explicit_start_idx = None
            explicit_end_idx = None
        if (
            explicit_start_idx is not None
            and explicit_end_idx is not None
            and 0 <= explicit_start_idx < len(rows)
            and 0 <= explicit_end_idx < len(rows)
        ):
            start_idx = int(explicit_start_idx)
            end_idx = int(explicit_end_idx)

        if start_idx is None or end_idx is None:
            process_start_line, process_start_point_no = self._parse_line_point_label(interval.get("process_start_label"))
            process_end_line, process_end_point_no = self._parse_line_point_label(interval.get("process_end_label"))
            if start_idx is None:
                start_idx = _pick_start(process_start_line, process_start_point_no)
            if end_idx is None:
                end_idx = _pick_end(process_end_line, process_end_point_no)

        if start_idx is None or end_idx is None:
            try:
                start_line = int(
                    interval.get("start_line")
                )
                end_line = int(
                    interval.get("end_line")
                )
            except Exception:
                start_line = None
                end_line = None
            if (
                start_line is not None
                and end_line is not None
                and _ensure_process_arrays()
            ):
                line_mask = (
                    (process_line_arr >= min(start_line, end_line))
                    & (process_line_arr <= max(start_line, end_line))
                )
                matching = np.flatnonzero(line_mask)
                if matching.size:
                    if start_idx is None:
                        start_idx = int(matching[0])
                    if end_idx is None:
                        end_idx = int(matching[-1])

        if start_idx is None or end_idx is None:
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        safe_start = max(0, min(int(start_idx), len(rows) - 1))
        safe_end = max(0, min(int(end_idx), len(rows) - 1))
        if safe_end < safe_start:
            return None

        start_row = rows[safe_start]
        end_row = rows[safe_end]
        start_line = self._get_process_row_sample_line(start_row, fallback=safe_start)
        end_line = self._get_process_row_sample_line(end_row, fallback=safe_end)
        try:
            start_point_idx = max(int(start_row.get("process_point_index", 0)), 0)
        except Exception:
            start_point_idx = 0
        try:
            end_point_idx = max(int(end_row.get("process_point_index", 0)), 0)
        except Exception:
            end_point_idx = 0

        try:
            start_s = float(start_row.get("path_start"))
        except Exception:
            start_s = float("nan")
        try:
            end_s = float(end_row.get("path_end"))
        except Exception:
            end_s = float("nan")

        formatter = getattr(self, "format_line_point", None)
        if callable(formatter):
            process_start_label = formatter(start_line, start_point_idx)
            process_end_label = formatter(end_line, end_point_idx)
        else:
            process_start_label = f"{int(start_line)}.{int(start_point_idx) + 1}"
            process_end_label = f"{int(end_line)}.{int(end_point_idx) + 1}"

        resolved = {
            "start_idx": int(safe_start),
            "end_idx": int(safe_end),
            "start_line": int(start_line),
            "end_line": int(end_line),
            "start_point_index": int(start_point_idx),
            "end_point_index": int(end_point_idx),
            "process_start_label": str(process_start_label),
            "process_end_label": str(process_end_label),
        }
        if np.isfinite(start_s):
            resolved["start_s"] = float(start_s)
        if np.isfinite(end_s):
            resolved["end_s"] = float(end_s)
        return resolved

    def _materialize_profile_pit_records(self, records):
        materialized = []
        if not records:
            return materialized

        for record in records:
            if not isinstance(record, dict):
                continue
            runtime_record = dict(record)
            has_runtime_bounds = any(
                runtime_record.get(key) is not None
                for key in (
                    "sample_start_idx",
                    "sample_end_idx",
                    "display_start_x",
                    "display_end_x",
                    "start_label",
                    "end_label",
                )
            )
            if has_runtime_bounds:
                current = runtime_record
            else:
                current = self._extract_interval_template_record(runtime_record)
            if not isinstance(current, dict):
                current = dict(runtime_record)

            process_bounds = self._resolve_interval_process_bounds(current)
            if process_bounds:
                for key, value in process_bounds.items():
                    current.setdefault(key, value)
                process_x_bounds = self._resolve_interval_process_x_bounds(current, process_bounds=process_bounds)
                if process_x_bounds:
                    for key, value in process_x_bounds.items():
                        current.setdefault(key, value)

            sample_bounds = self._resolve_interval_sample_bounds(current)
            if sample_bounds:
                for key, value in sample_bounds.items():
                    current.setdefault(key, value)

            start_label = str(current.get("start_label") or "").strip()
            end_label = str(current.get("end_label") or "").strip()
            if start_label or end_label:
                current["interval_range"] = (
                    f"{start_label}-{end_label}" if start_label and end_label else (start_label or end_label)
                )
            materialized.append(current)
        return materialized

    def _resolve_profile_segment_state_code(self, record):
        segment_type = str(record.get("segment_type") or "").strip().lower()
        fixed_codes = dict(STATE_CODE_BY_TYPE)
        fixed_codes["steady_cutting"] = fixed_codes["steady"]
        if segment_type in fixed_codes:
            return int(fixed_codes[segment_type])
        try:
            return int(record.get("state_code"))
        except Exception:
            steady_subtype = str(record.get("steady_subtype") or "").strip().lower()
            is_idle_interval = bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle"
            if segment_type == "nonsteady" or steady_subtype == "nonsteady":
                return 0
            if segment_type == "idle" or steady_subtype == "idle" or is_idle_interval:
                return 1
            return 2

    def _resolve_smif_state_code(self, record):
        """将六态记录投影到 SMIF 现有的非稳态/空载/稳态三种显示语义。"""
        if not isinstance(record, dict):
            return 0
        segment_type = str(record.get("segment_type") or "").strip().lower()
        if segment_type in {"steady", "steady_cutting"}:
            return 2
        if segment_type == "idle":
            return 1
        if segment_type in {"entry", "transition", "nonsteady", "exit"}:
            return 0
        state_code = int(self._resolve_profile_segment_state_code(record))
        return state_code if state_code in {0, 1, 2} else 0

    def _extract_profile_interval_records(self, profile=None, include_nonsteady=False):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return []

        if include_nonsteady:
            segment_records = self._extract_profile_segment_records(source_profile)
            if segment_records:
                return [dict(record) for record in segment_records if isinstance(record, dict)]

        raw_segment_records = [
            dict(record)
            for record in (source_profile.get("segment_records", []) or [])
            if isinstance(record, dict)
        ]
        if raw_segment_records:
            segment_records = self._extract_profile_segment_records(source_profile)
            materialized = self._materialize_profile_pit_records(segment_records)
            candidates = materialized if materialized else segment_records
            return self._get_steady_interval_records(candidates)

        def _materialize_steady_records(records):
            materialized = self._materialize_profile_pit_records(records)
            candidates = materialized if materialized else records
            return self._get_steady_interval_records(
                [dict(record) for record in candidates if isinstance(record, dict)]
            )

        raw_pit_records = [
            dict(record)
            for record in (source_profile.get("pit_records", []) or [])
            if isinstance(record, dict)
        ]
        if raw_pit_records:
            return _materialize_steady_records(raw_pit_records)

        raw_template_records = [
            dict(record)
            for record in (source_profile.get("interval_templates", []) or [])
            if isinstance(record, dict)
        ]
        if raw_template_records:
            return _materialize_steady_records(raw_template_records)
        return []

    def _extract_profile_segment_records(self, profile=None):
        if not isinstance(profile, dict):
            profile = getattr(self, "active_kc_profile", None)
        if not isinstance(profile, dict):
            return []
        has_process_data = bool(getattr(self, "data", None))

        records = profile.get("segment_records")
        if not isinstance(records, list) or not records:
            interval_records = self._extract_profile_interval_records(profile)
            if interval_records and has_process_data:
                return self._build_profile_segment_records(interval_records=interval_records)
            raw_templates = [
                dict(record)
                for record in (profile.get("interval_templates", []) or [])
                if isinstance(record, dict)
            ]
            materialized = self._materialize_profile_pit_records(raw_templates) if has_process_data else []
            if materialized and has_process_data:
                return self._build_profile_segment_records(interval_records=materialized)
            return []

        normalized = []
        for record in records:
            if not isinstance(record, dict):
                continue
            current = dict(record)
            process_bounds = self._resolve_interval_process_bounds(current)
            if process_bounds:
                for key, value in process_bounds.items():
                    current.setdefault(key, value)
                process_x_bounds = self._resolve_interval_process_x_bounds(current, process_bounds=process_bounds)
                if process_x_bounds:
                    for key, value in process_x_bounds.items():
                        current.setdefault(key, value)
            sample_bounds = self._resolve_interval_sample_bounds(current)
            if sample_bounds:
                for key, value in sample_bounds.items():
                    current.setdefault(key, value)
            current["state_code"] = int(self._resolve_profile_segment_state_code(current))
            normalized.append(current)
        return normalized

    def _record_represents_steady_interval(self, record):
        if not isinstance(record, dict):
            return False
        segment_type = str(record.get("segment_type") or "").strip().lower()
        is_idle_interval = bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle"
        if is_idle_interval:
            return False
        if segment_type:
            return segment_type in {"steady", "steady_cutting"}
        return int(self._resolve_profile_segment_state_code(record)) == 2

    def _refresh_interval_process_descriptors(self, record):
        """按当前工艺点刷新区间兼容描述值，不改动任何区间边界或状态字段。"""
        current = dict(record) if isinstance(record, dict) else {}
        rows = getattr(self, "data", None) or []
        if not current or not rows:
            return current
        try:
            start_idx = int(current.get("start_idx"))
            end_idx = int(current.get("end_idx"))
        except (TypeError, ValueError):
            return current
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= len(rows):
            return current
        interval_rows = [row for row in rows[start_idx:end_idx + 1] if isinstance(row, dict)]

        def _mean_value(*keys):
            values = []
            for row in interval_rows:
                value = None
                for key in keys:
                    if row.get(key) is not None:
                        value = row.get(key)
                        break
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            return float(np.mean(values)) if values else None

        descriptor_sources = {
            "a_p": ("ap", "a_p"),
            "a_e": ("ae", "a_e"),
            "F_plan": ("feed_effective", "F_program", "F_plan"),
            "p_idle": ("P_idle",),
            "p_pred": ("P",),
        }
        for target_key, source_keys in descriptor_sources.items():
            value = _mean_value(*source_keys)
            if value is not None:
                current[target_key] = value
        return current

    def _get_steady_interval_records(self, records=None):
        if isinstance(records, list):
            source_records = records
        else:
            segment_records = self._get_current_segment_records(allow_profile_fallback=False)
            source_records = (
                segment_records
                if segment_records
                else self._get_current_interval_records(allow_profile_fallback=False)
            )

        steady_records = []
        for record in source_records or []:
            if not isinstance(record, dict):
                continue
            current = dict(record)
            if not self._record_represents_steady_interval(current):
                continue
            refreshed = self._refresh_interval_process_descriptors(current)
            if refreshed:
                steady_records.append(refreshed)
        return steady_records

    def _has_authoritative_segmentation_state(self):
        return bool(
            getattr(self, "_current_interval_ready", False)
            and str(getattr(self, "_current_interval_source", "") or "") == "segmentation"
        )

    def _refresh_authoritative_segmentation_interval_descriptors(self):
        if not self._has_authoritative_segmentation_state() or not getattr(self, "data", None):
            return False
        interval_records = [
            self._refresh_interval_process_descriptors(record)
            for record in self._get_current_interval_records(allow_profile_fallback=False)
        ]
        return self._set_current_interval_state(
            interval_records=interval_records,
            segment_records=self._get_current_segment_records(allow_profile_fallback=False),
            point_kc_map=dict(getattr(self, "current_interval_point_kc_map", {}) or {}),
            source="segmentation",
            profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
            context_signature=str(getattr(self, "_current_interval_context_signature", "") or ""),
            prediction_source=str(getattr(self, "_current_interval_prediction_source", "no_profile") or "no_profile"),
            measurement_case_signature=str(
                getattr(self, "_current_interval_measurement_case_signature", "") or ""
            ),
        )

    def _profile_contains_steady_interval_records(self, profile=None, interval_records=None):
        if isinstance(interval_records, list) and interval_records:
            candidate_records = [dict(record) for record in interval_records if isinstance(record, dict)]
        else:
            source_profile = profile if isinstance(profile, dict) else getattr(self, "active_kc_profile", None)
            candidate_records = self._extract_profile_interval_records(source_profile)
        for record in candidate_records or []:
            if self._record_represents_steady_interval(record):
                return True
        source_profile = profile if isinstance(profile, dict) else getattr(self, "active_kc_profile", None)
        if isinstance(source_profile, dict):
            for record in (source_profile.get("segment_records", []) or []):
                if isinstance(record, dict) and self._record_represents_steady_interval(record):
                    return True
        return False

    def _select_interval_records_for_profile_persistence(self):
        return self._get_steady_interval_records()

    def _extract_interval_template_record(self, record, process_rows=None):
        if not isinstance(record, dict):
            return None

        current = dict(record)
        process_bounds = self._resolve_interval_process_bounds(current, process_rows=process_rows)
        runtime_keys = {
            "sample_start_idx",
            "sample_end_idx",
            "sample_start_line",
            "sample_end_line",
            "sample_start_label",
            "sample_end_label",
            "sample_anchor_start_idx",
            "sample_anchor_end_idx",
            "sample_anchor_start_label",
            "sample_anchor_end_label",
            "start_label",
            "end_label",
            "display_start_x",
            "display_end_x",
            "display_start_t",
            "display_end_t",
            "sample_count",
            "sample_kc_profile",
            "fit_df",
            "prediction_cache",
            "p_meas",
            "p_pred",
            "actual_load_std",
            "actual_load_diff_std",
            "valid_kc_count",
            "gated_out_count",
            "sigma_idle",
            "delta_mrr",
            "measurement_binding",
            "measurement_runtime",
            "sample_anchor_x",
            "process_anchor_x",
            "start_label",
            "end_label",
        }

        template_record = {}
        for key, value in current.items():
            key_text = str(key)
            if key_text in runtime_keys or key_text.startswith("sample_") or key_text.startswith("display_"):
                continue
            template_record[key_text] = value

        if process_bounds:
            template_record.update(process_bounds)
            process_x_bounds = self._resolve_interval_process_x_bounds(current, process_bounds=process_bounds, process_rows=process_rows)
            if process_x_bounds:
                template_record["process_start_x"] = process_x_bounds.get("process_start_x")
                template_record["process_end_x"] = process_x_bounds.get("process_end_x")
        elif not (
            template_record.get("start_idx") is not None
            and template_record.get("end_idx") is not None
        ) and not (
            template_record.get("process_start_label")
            or template_record.get("process_end_label")
            or template_record.get("start_line") is not None
            or template_record.get("end_line") is not None
        ):
            return None

        start_label = str(template_record.get("process_start_label") or "").strip()
        end_label = str(template_record.get("process_end_label") or "").strip()
        if start_label or end_label:
            template_record["interval_range"] = (
                f"{start_label}-{end_label}" if start_label and end_label else (start_label or end_label)
            )
        return template_record

    def _build_interval_templates_for_profile(self, interval_records=None):
        templates = []
        source_records = interval_records if isinstance(interval_records, list) else self._select_interval_records_for_profile_persistence()
        for record in source_records or []:
            template_record = self._extract_interval_template_record(record)
            if isinstance(template_record, dict):
                templates.append(template_record)
        return templates

    def _normalize_profile_line_kc_map(self, profile=None):
        source_profile = profile if isinstance(profile, dict) else self._get_saved_kc_profile_for_input()
        line_map = {}
        for raw_key, raw_value in (source_profile or {}).get("line_kc_map", {}).items():
            try:
                line_no = int(raw_key)
                kc_value = float(raw_value)
            except Exception:
                continue
            if np.isfinite(kc_value):
                line_map[line_no] = max(kc_value, 0.0)
        return line_map

    def _resolve_profile_point_kc_cap(self, profile=None):
        source_profile = profile if isinstance(profile, dict) else self._get_saved_kc_profile_for_input()
        if not isinstance(source_profile, dict):
            return float("inf")

        candidates = []
        for key in ("global_kc",):
            try:
                value = float(source_profile.get(key))
            except Exception:
                continue
            if np.isfinite(value) and value > 0.0:
                candidates.append(float(value))

        template_records = []
        for key in ("interval_templates", "pit_records"):
            template_records.extend(
                dict(record)
                for record in (source_profile.get(key, []) or [])
                if isinstance(record, dict)
            )

        for record in template_records:
            if not isinstance(record, dict):
                continue
            for key in ("K_c_UCB", "K_c_hat"):
                try:
                    value = float(record.get(key))
                except Exception:
                    continue
                if np.isfinite(value) and value > 0.0:
                    candidates.append(float(value))

        if candidates:
            upper = float(max(candidates))
            center = float(np.median(np.asarray(candidates, dtype=float)))
            return max(upper * 3.0, center * 4.0, 1.0)

        raw_point_values = []
        for raw_value in (source_profile.get("point_kc_map", {}) or {}).values():
            try:
                value = float(raw_value)
            except Exception:
                continue
            if np.isfinite(value) and value > 0.0:
                raw_point_values.append(float(value))
        if raw_point_values:
            point_arr = np.asarray(raw_point_values, dtype=float)
            p90 = float(np.percentile(point_arr, 90))
            center = float(np.median(point_arr))
            return max(p90 * 3.0, center * 6.0, 1.0)
        return float("inf")

    def _normalize_profile_point_kc_map(self, profile=None):
        source_profile = profile if isinstance(profile, dict) else self._get_saved_kc_profile_for_input()
        kc_cap = self._resolve_profile_point_kc_cap(source_profile)
        point_map = {}
        for raw_key, raw_value in (source_profile or {}).get("point_kc_map", {}).items():
            key_text = str(raw_key or "").strip()
            if not key_text:
                continue
            try:
                line_text, point_text = key_text.split(":", 1)
                line_no = int(line_text)
                point_idx = int(point_text)
                kc_value = float(raw_value)
            except Exception:
                continue
            if point_idx < 0:
                continue
            if np.isfinite(kc_value):
                point_map[(line_no, point_idx)] = min(max(kc_value, 0.0), kc_cap)
        return point_map

    def _profile_has_saved_payload(self, profile=None):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return False
        global_candidates = []
        for key in ("global_kc", "global_ke", "ke_value", "global_idle"):
            try:
                value = float(source_profile.get(key, float("nan")))
            except Exception:
                value = float("nan")
            global_candidates.append(value)
        return bool(
            source_profile.get("line_kc_map")
            or source_profile.get("point_kc_map")
            or source_profile.get("point_actual_feed_map")
            or source_profile.get("point_actual_mrr_map")
            or source_profile.get("interval_templates")
            or source_profile.get("pit_records")
            or source_profile.get("segment_records")
            or source_profile.get("idle_power_model")
            or any(np.isfinite(value) for value in global_candidates)
        )

    def _clip_nonnegative_numeric_array(self, values):
        array = np.asarray(values, dtype=float).copy()
        negative_mask = np.isfinite(array) & (array < 0.0)
        if np.any(negative_mask):
            array[negative_mask] = 0.0
        return array

    def _encode_compressed_numeric_array(self, values, dtype=np.float64):
        import base64
        import zlib

        array = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        raw_bytes = array.tobytes(order="C")
        compressed = zlib.compress(raw_bytes, level=6) if raw_bytes else b""
        return {
            "encoding": "base64+zlib",
            "dtype": str(array.dtype),
            "shape": [int(dim) for dim in array.shape],
            "data": base64.b64encode(compressed).decode("ascii") if compressed else "",
        }

    def _decode_compressed_numeric_array(self, payload, expected_size=None, fallback_dtype=np.float64):
        import base64
        import zlib

        if not isinstance(payload, dict):
            return np.asarray([], dtype=fallback_dtype)
        encoding = str(payload.get("encoding") or "").strip().lower()
        if encoding != "base64+zlib":
            return np.asarray([], dtype=fallback_dtype)
        dtype_name = str(payload.get("dtype") or np.dtype(fallback_dtype).name).strip() or np.dtype(fallback_dtype).name
        try:
            dtype = np.dtype(dtype_name)
        except Exception:
            dtype = np.dtype(fallback_dtype)
        shape = payload.get("shape") or []
        try:
            shape = tuple(int(dim) for dim in shape)
        except Exception:
            shape = ()
        data_text = str(payload.get("data") or "")
        if not data_text:
            array = np.asarray([], dtype=dtype)
        else:
            try:
                compressed = base64.b64decode(data_text.encode("ascii"))
                raw_bytes = zlib.decompress(compressed) if compressed else b""
                array = np.frombuffer(raw_bytes, dtype=dtype).copy()
            except Exception:
                return np.asarray([], dtype=fallback_dtype)
        if shape:
            try:
                array = array.reshape(shape)
            except Exception:
                return np.asarray([], dtype=fallback_dtype)
        if expected_size is not None and int(array.size) != int(expected_size):
            return np.asarray([], dtype=fallback_dtype)
        return array

    def _encode_compressed_bool_mask(self, values):
        import base64
        import zlib

        mask = np.ascontiguousarray(np.asarray(values, dtype=bool))
        packed = np.packbits(mask.astype(np.uint8), bitorder="little")
        raw_bytes = packed.tobytes(order="C")
        compressed = zlib.compress(raw_bytes, level=6) if raw_bytes else b""
        return {
            "encoding": "base64+zlib+packbits",
            "size": int(mask.size),
            "data": base64.b64encode(compressed).decode("ascii") if compressed else "",
        }

    def _decode_compressed_bool_mask(self, payload, expected_size=None):
        import base64
        import zlib

        if not isinstance(payload, dict):
            return np.zeros(0, dtype=bool)
        encoding = str(payload.get("encoding") or "").strip().lower()
        if encoding != "base64+zlib+packbits":
            return np.zeros(0, dtype=bool)
        try:
            bit_count = int(payload.get("size", 0) or 0)
        except Exception:
            bit_count = 0
        if expected_size is not None and bit_count != int(expected_size):
            return np.zeros(0, dtype=bool)
        data_text = str(payload.get("data") or "")
        if not data_text:
            return np.zeros(bit_count, dtype=bool)
        try:
            compressed = base64.b64decode(data_text.encode("ascii"))
            raw_bytes = zlib.decompress(compressed) if compressed else b""
            packed = np.frombuffer(raw_bytes, dtype=np.uint8)
            mask = np.unpackbits(packed, bitorder="little")[:bit_count].astype(bool, copy=False)
        except Exception:
            return np.zeros(0, dtype=bool)
        return mask

    def _build_manual_measurement_binding(self, measurement=None):
        import zlib

        payload = measurement if isinstance(measurement, dict) else getattr(self, "manual_measurement_data", None)
        if not isinstance(payload, dict):
            return {}

        sample_count = int(payload.get("sample_count", 0) or 0)
        program_line = np.asarray(payload.get("program_line", []), dtype=np.int32)
        actual_load = np.asarray(payload.get("actual_load", []), dtype=np.float32)
        if sample_count <= 0:
            sample_count = int(max(program_line.size, actual_load.size))
        if sample_count <= 0:
            return {}

        source_path = str(payload.get("source_file") or getattr(self, "manual_measurement_path", "") or "").strip()
        normalized_path = self._normalize_profile_binding_path(source_path)
        try:
            source_mtime = float(os.path.getmtime(source_path)) if source_path and os.path.exists(source_path) else 0.0
        except Exception:
            source_mtime = 0.0

        program_crc32 = ""
        if program_line.size:
            program_crc32 = f"{zlib.crc32(np.ascontiguousarray(program_line).view(np.uint8)) & 0xffffffff:08x}"
        actual_crc32 = ""
        if actual_load.size:
            actual_crc32 = f"{zlib.crc32(np.ascontiguousarray(actual_load).view(np.uint8)) & 0xffffffff:08x}"

        template_context = self._build_profile_template_context()
        return {
            "source_path": normalized_path,
            "source_mtime": round(float(source_mtime), 6),
            "sample_count": int(sample_count),
            "program_line_crc32": program_crc32,
            "actual_load_crc32": actual_crc32,
            "process_path": str(template_context.get("process_path") or ""),
            "process_hash": str(template_context.get("process_hash") or ""),
            "gcode_path": str(template_context.get("gcode_path") or ""),
            "gcode_hash": str(template_context.get("gcode_hash") or ""),
        }

    def _measurement_binding_matches(self, expected_binding, current_binding):
        if not isinstance(expected_binding, dict) or not isinstance(current_binding, dict):
            return False
        try:
            expected_count = int(expected_binding.get("sample_count", 0) or 0)
            current_count = int(current_binding.get("sample_count", 0) or 0)
        except Exception:
            return False
        if expected_count <= 0 or current_count <= 0 or expected_count != current_count:
            return False

        for key in ("program_line_crc32", "actual_load_crc32"):
            expected_value = str(expected_binding.get(key) or "").strip().lower()
            current_value = str(current_binding.get(key) or "").strip().lower()
            if expected_value and current_value and expected_value != current_value:
                return False

        expected_path = str(expected_binding.get("source_path") or "").strip()
        current_path = str(current_binding.get("source_path") or "").strip()
        if expected_path and current_path and expected_path != current_path:
            return False

        try:
            expected_mtime = float(expected_binding.get("source_mtime", 0.0) or 0.0)
            current_mtime = float(current_binding.get("source_mtime", 0.0) or 0.0)
        except Exception:
            expected_mtime = 0.0
            current_mtime = 0.0
        if expected_path and current_path and expected_mtime > 0.0 and current_mtime > 0.0 and abs(expected_mtime - current_mtime) > 1e-6:
            return False

        for key in ("process", "gcode"):
            path_key = f"{key}_path"
            hash_key = f"{key}_hash"
            expected_path_value = self._normalize_profile_binding_path(expected_binding.get(path_key))
            current_path_value = self._normalize_profile_binding_path(current_binding.get(path_key))
            expected_hash = str(expected_binding.get(hash_key) or "").strip().lower()
            current_hash = str(current_binding.get(hash_key) or "").strip().lower()
            if expected_hash and current_hash:
                if expected_hash != current_hash:
                    return False
                continue
            if expected_path_value and current_path_value and expected_path_value != current_path_value:
                return False
        return True

    def _profiles_share_interval_signature(self, left_profile, right_profile):
        if not isinstance(left_profile, dict) or not isinstance(right_profile, dict):
            return False

        left_records = self._extract_profile_interval_records(left_profile)
        right_records = self._extract_profile_interval_records(right_profile)
        if len(left_records) != len(right_records):
            return False
        if len((left_profile.get("point_kc_map") or {})) != len((right_profile.get("point_kc_map") or {})):
            return False

        def _signature(records):
            normalized = []
            for record in records:
                if not isinstance(record, dict):
                    continue
                kc_value = record.get("K_c_hat")
                try:
                    kc_value = float(kc_value)
                except Exception:
                    kc_value = float("nan")
                normalized.append(
                    (
                        str(record.get("process_start_label") or ""),
                        str(record.get("process_end_label") or ""),
                        round(kc_value, 6) if np.isfinite(kc_value) else None,
                    )
                )
            return tuple(normalized)

        return _signature(left_records) == _signature(right_records)

    def _restore_profile_measurement_binding_from_runtime(self, profile, measurement=None, process_path=None):
        source_profile = dict(profile) if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return profile
        if source_profile.get("measurement_binding") and source_profile.get("measurement_case_signature"):
            return source_profile

        runtime_profile = getattr(self, "runtime_identified_kc_profile", None)
        if not isinstance(runtime_profile, dict):
            return source_profile
        runtime_profile = self._normalize_loaded_kc_profile(runtime_profile)
        if not isinstance(runtime_profile, dict):
            return source_profile
        if not self._profile_matches_current_context(runtime_profile, process_path=process_path):
            return source_profile
        if not self._profiles_share_interval_signature(source_profile, runtime_profile):
            return source_profile

        restored = False
        runtime_binding = dict(runtime_profile.get("measurement_binding") or {})
        if runtime_binding and not source_profile.get("measurement_binding"):
            source_profile["measurement_binding"] = runtime_binding
            restored = True
        runtime_case_signature = str(
            runtime_profile.get("measurement_case_signature")
            or getattr(self, "runtime_identified_profile_case_signature", "")
            or self._get_current_measurement_case_signature(measurement)
            or ""
        ).strip()
        if runtime_case_signature and not source_profile.get("measurement_case_signature"):
            source_profile["measurement_case_signature"] = runtime_case_signature
            restored = True
        if restored:
            self._debug_prediction_state_event(
                "restore_profile_measurement_binding_from_runtime",
                kc_map_source="runtime_fit",
                measurement_case_signature=str(source_profile.get("measurement_case_signature") or "none"),
            )
        return source_profile

    def _profile_matches_current_measurement_binding(self, profile, measurement=None, process_path=None):
        source_profile = profile if isinstance(profile, dict) else None
        payload = measurement if isinstance(measurement, dict) else getattr(self, "manual_measurement_data", None)
        if not isinstance(source_profile, dict) or not isinstance(payload, dict):
            return False
        if str(getattr(self, "sample_data_mode", "") or "").strip() != "experiment_measurement":
            return False
        if not self._profile_matches_current_context(source_profile, process_path=process_path):
            return False

        current_binding = self._build_manual_measurement_binding(payload)
        expected_binding = dict(source_profile.get("measurement_binding") or {})
        if expected_binding and current_binding and self._measurement_binding_matches(expected_binding, current_binding):
            return True

        stored_case_signature = str(source_profile.get("measurement_case_signature") or "").strip()
        current_case_signature = self._get_current_measurement_case_signature(payload)
        return bool(stored_case_signature and current_case_signature and stored_case_signature == current_case_signature)

    def _promote_imported_profile_to_runtime_if_measurement_bound(self, profile, measurement=None, process_path=None, file_path=""):
        source_profile = self._restore_profile_measurement_binding_from_runtime(
            profile,
            measurement=measurement,
            process_path=process_path,
        )
        if not isinstance(source_profile, dict):
            return None
        if not self._profile_matches_current_measurement_binding(
            source_profile,
            measurement=measurement,
            process_path=process_path,
        ):
            return None

        resolved_file_path = str(
            file_path
            or getattr(self, "imported_kc_profile_path", "")
            or getattr(self, "active_kc_profile_path", "")
            or ""
        ).strip()
        self.imported_kc_profile = dict(source_profile)
        if resolved_file_path:
            self.imported_kc_profile_path = resolved_file_path
        promoted_profile = self._activate_profile_state(
            source_profile,
            origin="runtime_identified_profile",
            file_path=resolved_file_path,
            case_signature=self._get_current_measurement_case_signature(measurement),
        )
        self._debug_prediction_state_event(
            "promote_imported_profile_to_runtime",
            kc_map_source="runtime_fit",
            profile_path=os.path.basename(resolved_file_path) if resolved_file_path else "memory",
        )
        return promoted_profile

    def _build_sample_kc_profile_from_current_measurement(self):
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            return None
        measurement = getattr(self, "manual_measurement_data", None)
        if not isinstance(measurement, dict):
            return None

        sample_kc_values = self._clip_nonnegative_numeric_array(measurement.get("sample_kc_values", []))
        sample_kc_valid_mask = np.asarray(measurement.get("sample_kc_valid_mask", []), dtype=bool)
        if sample_kc_values.size == 0 or sample_kc_values.size != sample_kc_valid_mask.size:
            return None
        if not np.any(sample_kc_valid_mask):
            return None

        binding = self._build_manual_measurement_binding(measurement)
        if not binding:
            return None

        return {
            "binding": binding,
            "sample_count": int(sample_kc_values.size),
            "valid_count": int(np.sum(sample_kc_valid_mask)),
            "kc_values_blob": self._encode_compressed_numeric_array(sample_kc_values, dtype=np.float64),
            "valid_mask_blob": self._encode_compressed_bool_mask(sample_kc_valid_mask),
        }

    def _resolve_saved_sample_kc_profile(self, measurement=None, process_path=None):
        payload = measurement if isinstance(measurement, dict) else getattr(self, "manual_measurement_data", None)
        target_process_path = str(process_path or self._get_primary_input_file_or_empty() or "").strip()
        source_profile = getattr(self, "imported_kc_profile", None)
        if not isinstance(source_profile, dict):
            source_profile = getattr(self, "active_kc_profile", None)
        if not isinstance(source_profile, dict):
            return None
        sample_profile = source_profile.get("sample_kc_profile")
        if not isinstance(sample_profile, dict):
            return None
        if not self._profile_matches_current_context(source_profile, process_path=target_process_path or None):
            return None

        expected_binding = dict(sample_profile.get("binding") or {})
        current_binding = self._build_manual_measurement_binding(payload)
        if not expected_binding or not current_binding:
            return None
        if not self._measurement_binding_matches(expected_binding, current_binding):
            return None

        expected_size = int(sample_profile.get("sample_count", current_binding.get("sample_count", 0)) or 0)
        if expected_size <= 0:
            return None
        kc_values = self._decode_compressed_numeric_array(
            sample_profile.get("kc_values_blob"),
            expected_size=expected_size,
            fallback_dtype=np.float64,
        )
        valid_mask = self._decode_compressed_bool_mask(
            sample_profile.get("valid_mask_blob"),
            expected_size=expected_size,
        )
        if kc_values.size != expected_size or valid_mask.size != expected_size:
            return None
        return {
            "kc_values": self._clip_nonnegative_numeric_array(kc_values),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "source": "sample_kc_profile",
        }

    def _should_apply_saved_sample_profile_for_measurement_prediction(self):
        """实验实测在绑定匹配且未强制重算时，优先复用历史 sample_kc_profile。"""
        return bool(
            self._normalize_profile_origin(getattr(self, "profile_origin", "no_profile")) == "imported_profile"
            and hasattr(self, "_is_imported_profile_forward_lock_active")
            and self._is_imported_profile_forward_lock_active()
        )

    def _resolve_interval_records_for_measurement_prediction(self):
        records = self._get_steady_interval_records()
        if records and self._can_reuse_current_interval_template(
            prediction_source=self._get_prediction_source(),
            measurement=getattr(self, "manual_measurement_data", None),
        ):
            materialized = self._materialize_profile_pit_records(records)
            if materialized:
                return materialized
        _origin, profile = self._resolve_forward_prediction_profile(
            measurement=getattr(self, "manual_measurement_data", None),
            process_path=self._get_primary_input_file_or_empty(),
            allow_autoload_imported=self._should_allow_imported_profile_autoload(),
        )
        if isinstance(profile, dict):
            return self._get_steady_interval_records(
                self._extract_profile_interval_records(profile)
            )
        return []

    def _summarize_runtime_interval_from_sample_df(self, interval, sample_df):
        if not isinstance(interval, dict) or sample_df is None or sample_df.empty:
            return None

        def _series_to_float_array(column_name, default_value=np.nan):
            if column_name in sample_df.columns:
                source = sample_df[column_name]
            else:
                source = pd.Series(np.full(len(sample_df), default_value, dtype=float), index=sample_df.index)
            return pd.to_numeric(source, errors="coerce").to_numpy(dtype=float)

        row_count = int(len(sample_df))
        current = dict(interval)
        bounds = None
        try:
            explicit_start_idx = int(current.get("sample_start_idx"))
            explicit_end_idx = int(current.get("sample_end_idx"))
        except Exception:
            explicit_start_idx = None
            explicit_end_idx = None
        if (
            explicit_start_idx is not None
            and explicit_end_idx is not None
            and 0 <= explicit_start_idx < row_count
            and 0 <= explicit_end_idx < row_count
        ):
            bounds = {
                "sample_start_idx": int(min(explicit_start_idx, explicit_end_idx)),
                "sample_end_idx": int(max(explicit_start_idx, explicit_end_idx)),
            }
        else:
            bounds = self._resolve_interval_sample_bounds(current)
        if not bounds:
            return None
        current.update(bounds)

        start_idx = int(current.get("sample_start_idx"))
        end_idx = int(current.get("sample_end_idx"))
        if start_idx < 0 or end_idx < start_idx or start_idx >= row_count:
            return None
        end_idx = min(end_idx, row_count - 1)
        current["sample_start_idx"] = int(start_idx)
        current["sample_end_idx"] = int(end_idx)
        current["sample_count"] = int(end_idx - start_idx + 1)

        actual_load = _series_to_float_array("actual_load")
        predicted_load = _series_to_float_array("predicted_load")
        kc_point = _series_to_float_array("kc_point")
        sample_kc = _series_to_float_array("sample_kc")
        prediction_valid = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            if "prediction_valid" in sample_df.columns else np.ones(row_count, dtype=bool)
        )
        idle_point_mask = (
            sample_df["is_idle_point"].to_numpy(dtype=bool)
            if "is_idle_point" in sample_df.columns else np.zeros(row_count, dtype=bool)
        )
        kc_valid_mask = (
            sample_df["kc_valid"].to_numpy(dtype=bool)
            if "kc_valid" in sample_df.columns else np.isfinite(kc_point)
        )
        kc_gated_out_mask = (
            sample_df["kc_gated_out"].to_numpy(dtype=bool)
            if "kc_gated_out" in sample_df.columns else np.zeros(row_count, dtype=bool)
        )
        mrr_values = _series_to_float_array("mrr")
        sigma_idle = float(pd.to_numeric(sample_df.get("sigma_idle"), errors="coerce").iloc[0]) if "sigma_idle" in sample_df.columns and not sample_df.empty else 0.0
        delta_mrr = float(pd.to_numeric(sample_df.get("delta_mrr"), errors="coerce").iloc[0]) if "delta_mrr" in sample_df.columns and not sample_df.empty else 0.0

        interval_mask = np.zeros(row_count, dtype=bool)
        interval_mask[start_idx:end_idx + 1] = True
        is_idle_interval = bool(current.get("is_idle_interval")) or str(current.get("kc_source", "")).strip().lower() == "idle"

        actual_segment = actual_load[start_idx:end_idx + 1]
        finite_actual = actual_segment[np.isfinite(actual_segment)]
        if finite_actual.size:
            steady_stats = self._evaluate_measurement_steady_gate(
                finite_actual,
                sigma_idle=sigma_idle,
                sample_count=int(end_idx - start_idx + 1),
                min_sample_count=1,
            )
            current["actual_load_var"] = float(steady_stats.get("actual_load_var", 0.0))
            current["actual_load_std"] = float(steady_stats.get("actual_load_std", 0.0))
            current["actual_load_diff_std"] = float(steady_stats.get("actual_load_diff_std", 0.0))
            current["variance_limit"] = float(steady_stats.get("variance_limit", float("inf")))
            current["diff_std_limit"] = float(steady_stats.get("diff_std_limit", float("inf")))
            current["steady_pass"] = bool(steady_stats.get("steady_pass", current.get("steady_pass", True)))

        candidate_mask = (
            interval_mask
            & prediction_valid
            & (~idle_point_mask)
            & np.isfinite(mrr_values)
            & (mrr_values > 1e-12)
        )
        valid_interval_mask = candidate_mask & kc_valid_mask & np.isfinite(kc_point)
        valid_interval_kc = self._clip_nonnegative_numeric_array(kc_point[valid_interval_mask])
        if valid_interval_kc.size == 0:
            valid_interval_kc = self._clip_nonnegative_numeric_array(
                sample_kc[candidate_mask & np.isfinite(sample_kc)]
            )
        valid_kc_count = int(valid_interval_kc.size)
        gated_out_count = int(np.sum(interval_mask & kc_gated_out_mask))

        kc_hat = float("nan")
        sigma_kc = float("nan")
        if is_idle_interval:
            kc_hat = 0.0
            sigma_kc = 0.0
            current["kc_source"] = "idle"
        else:
            kc_hat, sigma_kc, _ = self._summarize_interval_kc_statistics(valid_interval_kc)
            if np.isfinite(kc_hat):
                current["kc_source"] = "measurement_mode"
            else:
                try:
                    kc_hat = float(current.get("K_c_hat"))
                except Exception:
                    kc_hat = float("nan")
                try:
                    sigma_kc = float(current.get("sigma_Kc"))
                except Exception:
                    sigma_kc = float("nan")

        current["valid_kc_count"] = int(valid_kc_count)
        current["gated_out_count"] = int(gated_out_count)
        current["sigma_idle"] = float(sigma_idle)
        current["delta_mrr"] = float(delta_mrr)

        p_meas_mask = interval_mask & np.isfinite(actual_load)
        if np.any(candidate_mask & np.isfinite(actual_load)):
            current["p_meas"] = float(np.mean(actual_load[candidate_mask & np.isfinite(actual_load)]))
        elif np.any(p_meas_mask):
            current["p_meas"] = float(np.mean(actual_load[p_meas_mask]))

        p_pred_mask = interval_mask & np.isfinite(predicted_load)
        if np.any(candidate_mask & np.isfinite(predicted_load)):
            current["p_pred"] = float(np.mean(predicted_load[candidate_mask & np.isfinite(predicted_load)]))
        elif np.any(p_pred_mask):
            current["p_pred"] = float(np.mean(predicted_load[p_pred_mask]))

        if np.isfinite(kc_hat):
            beta = 0.0
            try:
                beta = float(self.kc_beta.get())
            except Exception:
                beta = 0.0
            current["K_c_hat"] = max(float(kc_hat), 0.0)
            current["sigma_Kc"] = max(float(sigma_kc) if np.isfinite(sigma_kc) else 0.0, 0.0)
            current["K_c_UCB"] = float(current["K_c_hat"] + beta * current["sigma_Kc"])
        return current

    def _refresh_current_interval_runtime_state_from_sample_df(self, sample_df, interval_records=None):
        if self._is_imported_profile_forward_lock_active():
            return False
        if sample_df is None or sample_df.empty:
            return False

        source_records = (
            interval_records
            if isinstance(interval_records, list)
            else self._get_current_interval_records(allow_profile_fallback=False)
        )
        if not source_records:
            return False

        if getattr(self, "data", None):
            materialized_records = self._materialize_profile_pit_records(source_records)
        else:
            materialized_records = []
        if not materialized_records:
            materialized_records = [dict(record) for record in source_records if isinstance(record, dict)]

        runtime_records = []
        for record in materialized_records:
            runtime_record = self._summarize_runtime_interval_from_sample_df(record, sample_df)
            runtime_records.append(runtime_record if isinstance(runtime_record, dict) else dict(record))
        if not runtime_records:
            return False

        return bool(
            self._set_current_interval_state(
                interval_records=[dict(record) for record in runtime_records],
                segment_records=self._get_current_segment_records(allow_profile_fallback=False),
                point_kc_map=dict(getattr(self, "current_interval_point_kc_map", {}) or {}),
                source=str(getattr(self, "_current_interval_source", "") or ""),
                profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
                context_signature=str(getattr(self, "_current_interval_context_signature", "") or ""),
                prediction_source=str(getattr(self, "_current_interval_prediction_source", "no_profile") or "no_profile"),
                measurement_case_signature=str(getattr(self, "_current_interval_measurement_case_signature", "") or ""),
            )
        )

    def _apply_steady_interval_kc_to_sample_df(
        self,
        sample_df,
        intervals=None,
        *,
        authoritative_interval_kc=False,
        write_to_prediction=True,
        profile=None,
    ):
        if sample_df is None or sample_df.empty:
            return sample_df

        interval_records = intervals if isinstance(intervals, list) else self._resolve_interval_records_for_measurement_prediction()
        if not interval_records:
            return sample_df

        sample_df = self._initialize_measurement_prediction_channels(sample_df.copy())
        row_count = len(sample_df)
        predicted_kc = pd.to_numeric(sample_df.get("predicted_kc"), errors="coerce").to_numpy(dtype=float)
        predicted_load = pd.to_numeric(sample_df.get("predicted_load"), errors="coerce").to_numpy(dtype=float)
        predicted_source = (
            sample_df["predicted_kc_source"].astype(str).to_numpy(dtype=object)
            if "predicted_kc_source" in sample_df.columns else np.full(row_count, "", dtype=object)
        )
        interval_summary_kc = pd.to_numeric(sample_df.get("interval_summary_kc"), errors="coerce").to_numpy(dtype=float)
        interval_summary_load = pd.to_numeric(sample_df.get("interval_summary_load"), errors="coerce").to_numpy(dtype=float)
        interval_summary_source = (
            sample_df["interval_summary_source"].astype(str).to_numpy(dtype=object)
            if "interval_summary_source" in sample_df.columns else np.full(row_count, "", dtype=object)
        )
        display_kc = pd.to_numeric(sample_df.get("display_predicted_kc"), errors="coerce").to_numpy(dtype=float)
        display_load = pd.to_numeric(sample_df.get("display_predicted_load"), errors="coerce").to_numpy(dtype=float)
        display_source = (
            sample_df["display_prediction_source"].astype(str).to_numpy(dtype=object)
            if "display_prediction_source" in sample_df.columns
            else np.full(row_count, "", dtype=object)
        )
        kc_point = pd.to_numeric(sample_df.get("kc_point"), errors="coerce").to_numpy(dtype=float)
        sample_kc = pd.to_numeric(sample_df.get("sample_kc"), errors="coerce").to_numpy(dtype=float)
        prediction_valid = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            if "prediction_valid" in sample_df.columns else np.ones(row_count, dtype=bool)
        )
        kc_valid = (
            sample_df["kc_valid"].to_numpy(dtype=bool)
            if "kc_valid" in sample_df.columns else np.isfinite(kc_point)
        )
        sample_kc_valid = (
            sample_df["sample_kc_valid"].to_numpy(dtype=bool)
            if "sample_kc_valid" in sample_df.columns else np.isfinite(sample_kc)
        )
        idle_point_mask = (
            sample_df["is_idle_point"].to_numpy(dtype=bool)
            if "is_idle_point" in sample_df.columns else np.zeros(row_count, dtype=bool)
        )
        idle_power = pd.to_numeric(sample_df.get("idle_power"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ap_values = pd.to_numeric(sample_df.get("ap"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if "forward_prediction_mrr" in sample_df.columns:
            mrr_values = pd.to_numeric(sample_df.get("forward_prediction_mrr"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        elif "process_mrr" in sample_df.columns:
            mrr_values = pd.to_numeric(sample_df.get("process_mrr"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        else:
            mrr_values = pd.to_numeric(sample_df.get("mrr"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ke_value = (
            self._resolve_profile_ke_value(profile, default=self.get_ke_value())
            if isinstance(profile, dict)
            else float(self.get_ke_value())
        )
        self._debug_prediction_state_event(
            "apply_steady_interval_kc_to_sample_df",
            display_mode=self._get_measurement_display_mode(),
            interval_source=self._resolve_measurement_interval_source(interval_records, fallback="steady_interval_summary"),
            entered=True,
        )

        for idx, interval in enumerate(self._get_steady_interval_records(interval_records), 1):
            try:
                start_idx = int(interval.get("sample_start_idx"))
                end_idx = int(interval.get("sample_end_idx"))
            except Exception:
                bounds = self._resolve_interval_sample_bounds(interval)
                if not bounds:
                    continue
                start_idx = int(bounds["sample_start_idx"])
                end_idx = int(bounds["sample_end_idx"])
            if start_idx < 0 or end_idx < start_idx or start_idx >= row_count:
                continue
            end_idx = min(end_idx, row_count - 1)
            interval_mask = np.zeros(row_count, dtype=bool)
            interval_mask[start_idx:end_idx + 1] = True
            interval_id = str(interval.get("zone_id") or interval.get("interval_id") or f"Z{idx:03d}")
            state_code = self._resolve_smif_state_code(interval)
            segment_type = str(interval.get("segment_type") or "").strip().lower()
            steady_subtype = str(interval.get("steady_subtype") or "").strip().lower()
            is_idle_interval = bool(interval.get("is_idle_interval")) or str(interval.get("kc_source", "")).strip().lower() == "idle"
            if int(state_code) == 0 or segment_type == "nonsteady" or steady_subtype == "nonsteady":
                continue
            if is_idle_interval:
                predicted_kc[interval_mask] = np.nan
                predicted_load[interval_mask] = idle_power[interval_mask]
                predicted_source[interval_mask] = "idle"
                interval_summary_kc[interval_mask] = np.nan
                interval_summary_load[interval_mask] = idle_power[interval_mask]
                interval_summary_source[interval_mask] = "idle"
                if write_to_prediction:
                    display_kc[interval_mask] = np.nan
                    display_load[interval_mask] = idle_power[interval_mask]
                    display_source[interval_mask] = "idle"
                continue

            candidate_mask = interval_mask & prediction_valid & (~idle_point_mask)
            valid_interval_kc = self._clip_nonnegative_numeric_array(
                kc_point[candidate_mask & kc_valid & np.isfinite(kc_point)]
            )
            if valid_interval_kc.size == 0:
                valid_interval_kc = self._clip_nonnegative_numeric_array(
                    sample_kc[candidate_mask & sample_kc_valid & np.isfinite(sample_kc)]
                )
            if valid_interval_kc.size == 0:
                valid_interval_kc = self._clip_nonnegative_numeric_array(
                    interval_summary_kc[candidate_mask & np.isfinite(interval_summary_kc)]
                )
            runtime_kc_hat, _runtime_sigma_kc, _ = self._summarize_interval_kc_statistics(valid_interval_kc)
            try:
                saved_kc_hat = float(interval.get("K_c_hat"))
            except Exception:
                saved_kc_hat = float("nan")
            if authoritative_interval_kc:
                kc_hat = saved_kc_hat
                kc_source = "profile_interval_mode"
            else:
                kc_hat = runtime_kc_hat if np.isfinite(runtime_kc_hat) else saved_kc_hat
                kc_source = "measurement_mode" if np.isfinite(runtime_kc_hat) else str(interval.get("kc_source") or "")
            if not np.isfinite(kc_hat):
                continue
            kc_hat = max(float(kc_hat), 0.0)

            apply_mask = (
                interval_mask
                & prediction_valid
                & (~idle_point_mask)
                & np.isfinite(mrr_values)
                & np.isfinite(ap_values)
                & (mrr_values > 1e-12)
            )
            if not np.any(apply_mask):
                continue

            computed_load = (
                idle_power[apply_mask]
                + kc_hat * mrr_values[apply_mask]
                + ke_value * ap_values[apply_mask]
            )
            computed_load = np.maximum(computed_load, 0.0)
            predicted_kc[apply_mask] = kc_hat
            predicted_load[apply_mask] = computed_load
            predicted_source[apply_mask] = kc_source
            interval_summary_kc[apply_mask] = kc_hat
            interval_summary_load[apply_mask] = computed_load
            interval_summary_source[apply_mask] = kc_source
            if write_to_prediction:
                display_kc[apply_mask] = kc_hat
                display_load[apply_mask] = computed_load
                display_source[apply_mask] = kc_source

        sample_df["predicted_kc"] = predicted_kc
        sample_df["predicted_load"] = predicted_load
        sample_df["predicted_kc_source"] = predicted_source
        sample_df["interval_summary_kc"] = interval_summary_kc
        sample_df["interval_summary_load"] = interval_summary_load
        sample_df["interval_summary_source"] = interval_summary_source
        if write_to_prediction:
            sample_df["display_predicted_kc"] = display_kc
            sample_df["display_predicted_load"] = display_load
            sample_df["display_prediction_source"] = display_source
        return self._sync_display_prediction_aliases(sample_df)

    def _get_effective_ke_value_from_profile(self, profile=None, default=0.0):
        current_ke = self._parse_optional_float(self.ke_coeff.get())
        if current_ke is not None:
            return float(current_ke)
        saved_profile = profile if isinstance(profile, dict) else self._get_saved_kc_profile_for_input()
        try:
            profile_ke = float((saved_profile or {}).get("ke_value"))
        except Exception:
            profile_ke = float(default)
        if not np.isfinite(profile_ke):
            return max(float(default), 0.0)
        return max(float(profile_ke), 0.0)

    def _resolve_profile_ke_value(self, profile=None, default=0.0):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return max(float(default), 0.0)
        for key in ("ke_value", "global_ke"):
            try:
                value = float(source_profile.get(key))
            except Exception:
                continue
            if np.isfinite(value):
                return max(float(value), 0.0)
        return max(float(default), 0.0)

    def _get_effective_global_kc_from_profile(self, profile=None, default=0.0):
        current_kc = self._parse_optional_float(self.kc_coeff.get())
        if current_kc is not None:
            return float(current_kc)
        saved_profile = profile if isinstance(profile, dict) else self._get_saved_kc_profile_for_input()
        try:
            profile_kc = float((saved_profile or {}).get("global_kc"))
        except Exception:
            profile_kc = float(default)
        if not np.isfinite(profile_kc):
            return max(float(default), 0.0)
        return max(float(profile_kc), 0.0)

    def _resolve_profile_global_kc(self, profile=None, default=0.0):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return max(float(default), 0.0)
        try:
            value = float(source_profile.get("global_kc"))
        except Exception:
            value = float(default)
        if not np.isfinite(value):
            value = float(default)
        return max(float(value), 0.0)

    def _ensure_process_point_metadata(self):
        if not self.data:
            return
        cache_key = (id(self.data), len(self.data))
        if getattr(self, "_process_point_metadata_cache_key", None) == cache_key:
            return
        has_complete_metadata = True
        for row in self.data:
            try:
                point_idx = int(row.get("process_point_index", -1))
                point_count = int(row.get("process_point_count", 0))
            except Exception:
                has_complete_metadata = False
                break
            if point_idx < 0 or point_count <= 0:
                has_complete_metadata = False
                break
        if has_complete_metadata:
            self._process_point_metadata_cache_key = cache_key
            return

        def _row_group_key(row, row_index):
            raw_line = row.get("line_no_raw")
            if raw_line is None:
                raw_line = row.get("line_no_aligned")
            if raw_line is None:
                return int(row_index)
            try:
                return int(raw_line)
            except Exception:
                return int(row_index)

        point_counts = {}
        for row_index, row in enumerate(self.data):
            raw_key = _row_group_key(row, row_index)
            point_counts[raw_key] = point_counts.get(raw_key, 0) + 1

        point_offsets = {}
        for row_index, row in enumerate(self.data):
            raw_key = _row_group_key(row, row_index)
            point_idx = int(point_offsets.get(raw_key, 0))
            point_offsets[raw_key] = point_idx + 1
            row["process_point_index"] = point_idx
            row["process_point_count"] = int(point_counts.get(raw_key, point_idx + 1))
        self._process_point_metadata_cache_key = cache_key

    def _build_full_point_kc_map_from_current_state(
        self,
        allow_profile_fallback=True,
        prefer_current_state=True,
        allow_measurement_point_fallback=False,
    ):
        if (
            prefer_current_state
            and bool(getattr(self, "_current_interval_ready", False))
            and self._can_reuse_current_interval_template(
                prediction_source=self._get_prediction_source(),
                measurement=getattr(self, "manual_measurement_data", None),
            )
        ):
            return dict(getattr(self, "current_interval_point_kc_map", {}) or {})
        if not self.data:
            return {}

        self._ensure_process_point_metadata()
        default_kc = self.get_kc_value()
        point_kc_map = {}
        source_profile = None
        if allow_profile_fallback:
            _origin, source_profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload(),
            )
        if isinstance(source_profile, dict):
            default_kc = self._resolve_profile_global_kc(source_profile, default=default_kc)
        profile_point_kc_map = (
            self._normalize_profile_point_kc_map(source_profile)
            if allow_profile_fallback and hasattr(self, "_normalize_profile_point_kc_map")
            else {}
        )
        profile_line_kc_map = (
            self._normalize_profile_line_kc_map(source_profile)
            if allow_profile_fallback and hasattr(self, "_normalize_profile_line_kc_map")
            else {}
        )
        use_profile_kc = bool(profile_point_kc_map or profile_line_kc_map)

        for idx, row in enumerate(self.data):
            try:
                line_no = int(row.get("line_no_aligned", idx))
            except Exception:
                line_no = int(idx)
            try:
                point_idx = int(row.get("process_point_index", 0))
            except Exception:
                point_idx = 0
            if use_profile_kc and (int(line_no), int(point_idx)) in profile_point_kc_map:
                kc_value = float(profile_point_kc_map[(int(line_no), int(point_idx))])
            elif use_profile_kc and int(line_no) in profile_line_kc_map:
                kc_value = float(profile_line_kc_map[int(line_no)])
            else:
                try:
                    kc_value = float(row.get("K_c", row.get("K", default_kc)))
                except Exception:
                    kc_value = float(default_kc)
            if np.isfinite(kc_value):
                point_kc_map[(int(line_no), int(point_idx))] = max(float(kc_value), 0.0)

        measurement = getattr(self, "manual_measurement_data", None)
        if allow_measurement_point_fallback and measurement and not use_profile_kc:
            aligned_lines = np.asarray(measurement.get("line_no_aligned", []), dtype=int)
            point_indices = np.asarray(measurement.get("process_point_index", []), dtype=int)
            kc_points = self._clip_nonnegative_numeric_array(
                measurement.get("kc_point", measurement.get("sample_kc_values", []))
            )
            kc_valid_mask = np.asarray(
                measurement.get("kc_valid_mask", measurement.get("sample_kc_valid_mask", [])),
                dtype=bool,
            )
            if (
                aligned_lines.size
                == point_indices.size
                == kc_points.size
                == kc_valid_mask.size
                and aligned_lines.size > 0
            ):
                grouped_values = {}
                for line_no, point_idx, kc_value, is_valid in zip(aligned_lines, point_indices, kc_points, kc_valid_mask):
                    if not is_valid or point_idx < 0 or not np.isfinite(kc_value):
                        continue
                    grouped_values.setdefault((int(line_no), int(point_idx)), []).append(float(kc_value))
                for point_key, values in grouped_values.items():
                    if values:
                        kc_hat, _sigma_kc, valid_values = self._summarize_interval_kc_statistics(values)
                        if np.isfinite(kc_hat) and len(valid_values) > 0:
                            point_kc_map[point_key] = max(float(kc_hat), 0.0)

        for idx, row in enumerate(self.data):
            try:
                line_no = int(row.get("line_no_aligned", idx))
            except Exception:
                line_no = int(idx)
            try:
                point_idx = int(row.get("process_point_index", 0))
            except Exception:
                point_idx = 0
            point_key = (int(line_no), int(point_idx))
            if point_key not in point_kc_map:
                point_kc_map[point_key] = float(default_kc)
        return point_kc_map

    def _build_point_kc_map_from_interval_records(self, interval_records, base_point_kc_map=None):
        if not self.data or not isinstance(interval_records, list):
            return dict(base_point_kc_map or {})

        self._ensure_process_point_metadata()
        point_kc_map = dict(base_point_kc_map or {})
        default_kc = float(self.get_kc_value())
        for interval in self._get_steady_interval_records(interval_records):
            process_bounds = self._resolve_interval_process_bounds(interval)
            if not process_bounds:
                continue
            start_idx = int(process_bounds.get("start_idx", -1))
            end_idx = int(process_bounds.get("end_idx", -1))
            if start_idx < 0 or end_idx < start_idx:
                continue
            end_idx = min(end_idx, len(self.data) - 1)
            is_idle_interval = bool(interval.get("is_idle_interval")) or str(interval.get("kc_source", "")).strip().lower() == "idle"
            try:
                kc_value = 0.0 if is_idle_interval else float(interval.get("K_c_hat"))
            except Exception:
                kc_value = float("nan")
            if not is_idle_interval and not np.isfinite(kc_value):
                kc_value = float(default_kc)
            kc_value = max(float(kc_value), 0.0) if np.isfinite(kc_value) else float(default_kc)
            for row_idx in range(start_idx, end_idx + 1):
                row = self.data[row_idx]
                try:
                    line_no = int(row.get("line_no_aligned", row_idx))
                except Exception:
                    line_no = int(row_idx)
                try:
                    point_idx = int(row.get("process_point_index", 0))
                except Exception:
                    point_idx = 0
                point_kc_map[(int(line_no), int(point_idx))] = float(kc_value)
        return point_kc_map

    def _build_profile_segment_records(self, interval_records=None):
        if not self.data:
            return []

        self._ensure_process_point_metadata()
        normalized_intervals = self._materialize_profile_pit_records(
            interval_records
            if isinstance(interval_records, list)
            else self._get_current_interval_records(allow_profile_fallback=False)
        )

        total_rows = int(len(self.data))
        row_indices = np.arange(total_rows, dtype=int)
        row_states = np.ones(total_rows, dtype=np.int8)
        interval_lookup = {}

        for record in normalized_intervals:
            if not isinstance(record, dict):
                continue
            process_bounds = self._resolve_interval_process_bounds(record)
            if not process_bounds:
                continue
            safe_start = int(process_bounds["start_idx"])
            safe_end = int(process_bounds["end_idx"])
            if safe_end < safe_start:
                safe_start, safe_end = safe_end, safe_start
            is_idle_interval = bool(record.get("is_idle_interval")) or str(record.get("kc_source", "")).strip().lower() == "idle"
            is_steady_interval = bool(record.get("steady_pass", True))
            state_code = 1 if is_idle_interval else (2 if is_steady_interval else 0)
            row_states[safe_start:safe_end + 1] = int(state_code)
            normalized_record = dict(record)
            normalized_record.update(process_bounds)
            interval_lookup[(safe_start, safe_end, state_code)] = normalized_record

        for row_idx, row in enumerate(self.data):
            if int(row_states[row_idx]) == 2:
                continue
            row_states[row_idx] = 0 if self._is_smif_process_row_cutting(row) else 1

        segment_records = []
        for segment_idx, (block_start, block_end, block_state) in enumerate(
            self._collect_smif_path_blocks(row_indices, row_states),
            1,
        ):
            start_row = self.data[int(block_start)]
            end_row = self.data[int(block_end)]
            start_line = self._get_process_row_sample_line(start_row, fallback=block_start)
            end_line = self._get_process_row_sample_line(end_row, fallback=block_end)
            try:
                start_point_idx = int(start_row.get("process_point_index", 0))
            except Exception:
                start_point_idx = 0
            try:
                end_point_idx = int(end_row.get("process_point_index", 0))
            except Exception:
                end_point_idx = 0
            try:
                start_s = float(start_row.get("path_start"))
            except Exception:
                start_s = float("nan")
            try:
                end_s = float(end_row.get("path_end"))
            except Exception:
                end_s = float("nan")

            if int(block_state) == 2:
                segment_type = "steady_cutting"
                steady_subtype = "cutting"
            elif int(block_state) == 1:
                segment_type = "idle"
                steady_subtype = "idle"
            else:
                segment_type = "nonsteady"
                steady_subtype = "nonsteady"

            formatter = getattr(self, "format_line_point", None)
            if callable(formatter):
                process_start_label = formatter(start_line, start_point_idx)
                process_end_label = formatter(end_line, end_point_idx)
            else:
                process_start_label = f"{int(start_line)}.{int(start_point_idx) + 1}"
                process_end_label = f"{int(end_line)}.{int(end_point_idx) + 1}"

            process_bounds = {
                "start_idx": int(block_start),
                "end_idx": int(block_end),
                "start_line": int(start_line),
                "end_line": int(end_line),
                "start_point_index": int(start_point_idx),
                "end_point_index": int(end_point_idx),
                "process_start_label": str(process_start_label),
                "process_end_label": str(process_end_label),
            }
            segment_record = {
                "segment_id": f"SEG{segment_idx:03d}",
                "state_code": int(block_state),
                "segment_type": str(segment_type),
                "steady_subtype": str(steady_subtype),
                "is_idle_interval": bool(int(block_state) == 1),
                **process_bounds,
                "sample_count": int(block_end - block_start + 1),
            }
            if np.isfinite(start_s):
                segment_record["start_s"] = float(start_s)
            if np.isfinite(end_s):
                segment_record["end_s"] = float(end_s)
            process_x_bounds = self._resolve_interval_process_x_bounds(segment_record, process_bounds=process_bounds)
            if process_x_bounds:
                segment_record.update(process_x_bounds)

            interval_match = interval_lookup.get((int(block_start), int(block_end), int(block_state)))
            if isinstance(interval_match, dict):
                merged_record = dict(interval_match)
                merged_record.update(segment_record)
                merged_record["segment_type"] = str(segment_type)
                merged_record["steady_subtype"] = str(steady_subtype)
                merged_record["is_idle_interval"] = bool(int(block_state) == 1)
                segment_record = merged_record
            if int(block_state) == 0:
                for key in ("K_c_hat", "sigma_Kc", "K_c_UCB"):
                    segment_record.pop(key, None)
                segment_record["kc_source"] = str(segment_record.get("kc_source") or "")
            elif int(block_state) == 1:
                segment_record["kc_source"] = "idle"
                segment_record["is_idle_interval"] = True

            start_label = str(segment_record.get("process_start_label") or "").strip()
            end_label = str(segment_record.get("process_end_label") or "").strip()
            if start_label or end_label:
                segment_record["interval_range"] = (
                    f"{start_label}-{end_label}" if start_label and end_label else (start_label or end_label)
            )
            segment_records.append(segment_record)
        return segment_records

    def _build_profile_point_kc_map_for_segments(self, segment_records, steady_records=None):
        measurement = getattr(self, "manual_measurement_data", None)
        if not isinstance(measurement, dict):
            return {}

        candidate_records = steady_records if isinstance(steady_records, list) else segment_records
        if not isinstance(candidate_records, list):
            return {}
        # profile 点级 Kc 仅使用完整稳态区间，其余五态只用于展示与追溯。
        selected_steady_records = self._get_steady_interval_records(candidate_records)
        if not selected_steady_records:
            return {}

        aligned_lines = np.asarray(measurement.get("line_no_aligned", []), dtype=int)
        point_indices = np.asarray(measurement.get("process_point_index", []), dtype=int)
        process_row_indices = np.asarray(measurement.get("process_row_index", []), dtype=int)
        kc_point = self._clip_nonnegative_numeric_array(measurement.get("kc_point", []))
        kc_valid = np.asarray(measurement.get("kc_valid_mask", []), dtype=bool)
        sample_kc = self._clip_nonnegative_numeric_array(measurement.get("sample_kc_values", []))
        sample_kc_valid = np.asarray(measurement.get("sample_kc_valid_mask", []), dtype=bool)
        expected_size = aligned_lines.size
        if (
            expected_size <= 0
            or point_indices.size != expected_size
            or process_row_indices.size != expected_size
            or kc_point.size != expected_size
            or kc_valid.size != expected_size
            or sample_kc.size != expected_size
            or sample_kc_valid.size != expected_size
        ):
            return {}

        # 区间处于 process row 坐标。排序区间端点后，一次性计算每个实测
        # 样本被多少个稳态区间覆盖；既保留重叠区间的重复计权语义，也避免
        # 每个区间都扫描全部实测点。
        interval_starts = []
        interval_ends = []
        for segment in selected_steady_records:
            if not isinstance(segment, dict):
                continue
            try:
                start_idx = int(segment.get("start_idx"))
                end_idx = int(segment.get("end_idx"))
            except Exception:
                continue
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            interval_starts.append(int(start_idx))
            interval_ends.append(int(end_idx))
        if not interval_starts:
            return {}

        sorted_starts = np.sort(np.asarray(interval_starts, dtype=int))
        sorted_ends = np.sort(np.asarray(interval_ends, dtype=int))
        sample_coverage = (
            np.searchsorted(sorted_starts, process_row_indices, side="right")
            - np.searchsorted(sorted_ends, process_row_indices, side="left")
        )

        preferred_kc = kc_valid & np.isfinite(kc_point)
        fallback_kc = (~preferred_kc) & sample_kc_valid & np.isfinite(sample_kc)
        usable = (
            (sample_coverage > 0)
            & (point_indices >= 0)
            & (preferred_kc | fallback_kc)
        )
        usable_indices = np.flatnonzero(usable)
        if usable_indices.size == 0:
            return {}

        kc_values = np.where(preferred_kc, kc_point, sample_kc)
        repeat_counts = sample_coverage[usable_indices].astype(int)
        expanded_indices = np.repeat(usable_indices, repeat_counts)
        grouped_frame = pd.DataFrame(
            {
                "line_no": aligned_lines[expanded_indices],
                "point_idx": point_indices[expanded_indices],
                "kc_value": kc_values[expanded_indices],
            }
        )
        grouped_frame["rounded_kc"] = np.round(
            grouped_frame["kc_value"].to_numpy(dtype=float),
            6,
        )
        group_columns = ["line_no", "point_idx"]
        centers = (
            grouped_frame.groupby(group_columns, sort=True)["kc_value"]
            .median()
            .rename("center")
        )
        candidate_counts = (
            grouped_frame.groupby(
                [*group_columns, "rounded_kc"],
                sort=True,
            )
            .size()
            .rename("count")
            .reset_index()
        )
        max_counts = candidate_counts.groupby(group_columns, sort=False)["count"].transform("max")
        candidates = candidate_counts.loc[candidate_counts["count"] == max_counts].copy()
        candidates = candidates.join(centers, on=group_columns)
        candidates["distance"] = np.abs(
            candidates["rounded_kc"].to_numpy(dtype=float)
            - candidates["center"].to_numpy(dtype=float)
        )
        winners = (
            candidates.sort_values(
                [*group_columns, "distance", "rounded_kc"],
                kind="mergesort",
            )
            .drop_duplicates(group_columns, keep="first")
        )
        return {
            f"{int(row.line_no)}:{int(row.point_idx)}": max(float(row.rounded_kc), 0.0)
            for row in winners.itertuples(index=False)
        }

    def _build_current_kc_profile_snapshot(self, source="measurement"):
        process_path = self.get_primary_input_file()
        if not process_path or not os.path.exists(process_path) or not self.data:
            return None

        current_interval_records = self._materialize_profile_pit_records(
            self._get_current_interval_records(allow_profile_fallback=False)
        )
        if not current_interval_records:
            current_interval_records = [
                dict(record)
                for record in self._get_current_interval_records(allow_profile_fallback=False)
                if isinstance(record, dict)
            ]
        steady_pit_records = self._materialize_profile_pit_records(
            self._get_steady_interval_records()
        )
        interval_templates = self._build_interval_templates_for_profile(steady_pit_records)
        if self._has_authoritative_segmentation_state():
            segment_records = self._get_current_segment_records(allow_profile_fallback=False)
        else:
            segment_records = self._build_profile_segment_records(interval_records=current_interval_records)
        point_kc_map = self._build_profile_point_kc_map_for_segments(
            segment_records,
            steady_records=steady_pit_records,
        )
        sample_kc_profile = self._build_sample_kc_profile_from_current_measurement()
        gcode_path = (
            str(self.gcode_nc_path_var.get().strip())
            if hasattr(getattr(self, "gcode_nc_path_var", None), "get")
            else ""
        )
        template_context = self._build_profile_template_context(process_path=process_path, gcode_path=gcode_path)
        profile = {
            "process_path": os.path.abspath(process_path),
            "gcode_path": gcode_path,
            "updated_at": datetime.now().isoformat(),
            "source": str(source or ""),
            "template_schema_version": 3,
            "template_context": template_context,
            "global_kc": float(self.get_kc_value()),
            "kc_sigma": float(self.kc_sigma.get()),
            "global_ke": float(self._get_effective_ke_value_from_profile(default=0.0)),
            "ke_value": float(self._get_effective_ke_value_from_profile(default=0.0)),
            "global_idle": float(self.p_idle_var.get() or 0.0),
            "idle_power_model": dict(getattr(self, "idle_power_model", {}) or {}),
            "idle_model_signature": str(getattr(self, "idle_model_signature", "") or ""),
            "interval_count": int(len(steady_pit_records)),
            "segment_count": int(len(segment_records)),
            "point_kc_count": int(len(point_kc_map)),
            "point_kc_map": dict(point_kc_map),
            "interval_templates": self._serialize_record_list(interval_templates),
            "pit_records": self._serialize_record_list(steady_pit_records),
            "segment_records": self._serialize_record_list(segment_records),
        }
        if isinstance(sample_kc_profile, dict):
            profile["sample_kc_profile"] = dict(sample_kc_profile)
        return self._normalize_loaded_kc_profile(profile)

    def _persist_current_kc_profile(self, source="measurement"):
        profile = self._build_current_kc_profile_snapshot(source=source)
        if not isinstance(profile, dict):
            return False

        measurement_case_signature = self._get_current_measurement_case_signature()
        self._activate_profile_state(
            profile,
            origin="runtime_identified_profile",
            file_path="",
            case_signature=measurement_case_signature,
            normalize=False,
        )
        self._sync_current_interval_state_prediction_context(
            prediction_source="runtime_identified_profile",
            measurement=getattr(self, "manual_measurement_data", None),
        )
        if hasattr(self, "kc_profile_status_var"):
            self.kc_profile_status_var.set("案例配置: 当前运行时辨识结果（未保存）")
        self.refresh_mechanism_status_summary()
        self._persist_app_config()
        self._debug_prediction_state_event(
            "persist_runtime_profile",
            measurement_case_signature=measurement_case_signature or "none",
            kc_map_source="runtime_fit",
        )
        return True

    def _apply_point_kc_map_to_current_data(
        self,
        point_kc_map=None,
        *,
        line_kc_map=None,
        ke_value=None,
        clear_interval_ids=True,
    ):
        if not self.data:
            return False

        self._ensure_process_point_metadata()
        resolved_point_kc_map = dict(point_kc_map or {})
        resolved_line_kc_map = dict(line_kc_map or {})
        resolved_ke_value = (
            self._get_effective_ke_value_from_profile(default=self.get_ke_value())
            if ke_value is None
            else max(float(ke_value), 0.0)
        )
        idle_predictor = self._create_idle_power_predictor() if hasattr(self, "_create_idle_power_predictor") else self.predict_idle_power

        spindle_speeds = []
        aligned_line_numbers = []
        process_point_indices = []
        for idx, row in enumerate(self.data):
            try:
                aligned_line_numbers.append(int(row.get("line_no_aligned", idx)))
            except Exception:
                aligned_line_numbers.append(int(idx))
            try:
                process_point_indices.append(int(row.get("process_point_index", 0)))
            except Exception:
                process_point_indices.append(0)
            try:
                spindle_speeds.append(float(row.get("S", self.current_program_speed.get() if hasattr(self, "current_program_speed") else 0.0) or 0.0))
            except Exception:
                spindle_speeds.append(0.0)

        speed_arr = np.asarray(spindle_speeds, dtype=float)
        rounded_speeds = np.round(speed_arr, 6)
        unique_speeds, inverse_indices = np.unique(rounded_speeds, return_inverse=True)
        unique_idle_powers = np.asarray([idle_predictor(speed) for speed in unique_speeds], dtype=float)
        idle_power_arr = unique_idle_powers[inverse_indices]

        default_kc = float(self.get_kc_value())
        for idx, row in enumerate(self.data):
            line_no = aligned_line_numbers[idx]
            spindle_speed = float(speed_arr[idx]) if idx < len(speed_arr) else 0.0
            idle_power = float(idle_power_arr[idx]) if idx < len(idle_power_arr) else 0.0
            point_idx = int(process_point_indices[idx]) if idx < len(process_point_indices) else 0
            local_kc = None
            if (int(line_no), int(point_idx)) in resolved_point_kc_map:
                local_kc = float(resolved_point_kc_map[(int(line_no), int(point_idx))])
            elif int(line_no) in resolved_line_kc_map:
                local_kc = float(resolved_line_kc_map[int(line_no)])
            else:
                try:
                    local_kc = float(row.get("K_c", default_kc))
                except Exception:
                    local_kc = float(default_kc)
            local_kc = max(float(local_kc), 0.0)
            ap_val = float(row.get("ap", 0.0) or 0.0)
            mrr_val = float(row.get("MRR", 0.0) or 0.0)
            total_power = max(idle_power + local_kc * mrr_val + resolved_ke_value * ap_val, 0.0)
            row["K"] = float(local_kc)
            row["K_c"] = float(local_kc)
            row["K_e"] = float(resolved_ke_value)
            row["P_idle"] = float(idle_power)
            row["P_edge"] = float(resolved_ke_value * ap_val)
            row["P"] = float(total_power)
            if clear_interval_ids:
                row.pop("steady_interval_id", None)
            try:
                angular_velocity = 2 * math.pi * spindle_speed / 60.0
                row["T"] = float(total_power / angular_velocity) if angular_velocity > 1e-9 else 0.0
            except Exception:
                row["T"] = 0.0
        self._invalidate_process_alignment_caches(reason="apply_point_kc_map")
        self._refresh_authoritative_segmentation_interval_descriptors()
        return True

    @staticmethod
    def _measurement_content_binding_matches(expected_binding, current_binding):
        """忽略文件位置，只按采样内容判断是否为同一份实际负载。"""
        if not isinstance(expected_binding, dict) or not isinstance(current_binding, dict):
            return False
        try:
            expected_count = int(expected_binding.get("sample_count", 0) or 0)
            current_count = int(current_binding.get("sample_count", 0) or 0)
        except Exception:
            return False
        if expected_count <= 0 or expected_count != current_count:
            return False
        expected_actual = str(expected_binding.get("actual_load_crc32") or "").strip().lower()
        current_actual = str(current_binding.get("actual_load_crc32") or "").strip().lower()
        if not expected_actual or not current_actual or expected_actual != current_actual:
            return False
        expected_lines = str(expected_binding.get("program_line_crc32") or "").strip().lower()
        current_lines = str(current_binding.get("program_line_crc32") or "").strip().lower()
        return not (
            expected_lines
            and current_lines
            and expected_lines != current_lines
        )

    def _profile_is_independent_from_current_measurement(
        self,
        profile=None,
        measurement=None,
    ):
        """判断前向 profile 是否独立于当前实际负载。"""
        source_profile = profile if isinstance(profile, dict) else None
        payload = (
            measurement
            if isinstance(measurement, dict)
            else getattr(self, "manual_measurement_data", None)
        )
        if not isinstance(source_profile, dict) or not isinstance(payload, dict):
            return True
        if str(getattr(self, "sample_data_mode", "") or "").strip() != "experiment_measurement":
            return True

        current_binding = self._build_manual_measurement_binding(payload)
        candidate_bindings = []
        top_level_binding = source_profile.get("measurement_binding")
        if isinstance(top_level_binding, dict):
            candidate_bindings.append(top_level_binding)
        sample_profile = source_profile.get("sample_kc_profile")
        if isinstance(sample_profile, dict) and isinstance(sample_profile.get("binding"), dict):
            candidate_bindings.append(sample_profile["binding"])
        if any(
            self._measurement_content_binding_matches(binding, current_binding)
            for binding in candidate_bindings
        ):
            return False

        stored_case_signature = str(
            source_profile.get("measurement_case_signature") or ""
        ).strip()
        current_case_signature = self._get_current_measurement_case_signature(payload)
        if (
            stored_case_signature
            and current_case_signature
            and stored_case_signature == current_case_signature
        ):
            return False

        source = str(source_profile.get("source") or "").strip().lower()
        def _binding_is_verifiable(binding):
            try:
                sample_count = int(binding.get("sample_count", 0) or 0)
            except Exception:
                return False
            return bool(
                sample_count > 0
                and str(binding.get("actual_load_crc32") or "").strip()
            )

        has_verifiable_training_binding = any(
            _binding_is_verifiable(binding)
            for binding in candidate_bindings
        )
        if source.startswith("measurement") and not has_verifiable_training_binding:
            # 无训练数据绑定的实测辨识 profile 无法证明与当前验证样本独立。
            return False
        return True

    def _apply_profile_prediction_parameters(self, profile, source_identity=""):
        """只激活前向预测参数，不导入 profile 中保存的历史六态区间。"""

        if not isinstance(profile, dict):
            return False

        idle_model = profile.get("idle_power_model")
        if isinstance(idle_model, dict) and idle_model.get("speeds") and idle_model.get("powers"):
            self.idle_power_model = dict(idle_model)
            self.idle_model_signature = str(
                profile.get("idle_model_signature")
                or source_identity
                or self._build_stable_prediction_digest(idle_model)
            )
        elif "idle_power_model" in profile:
            self.idle_power_model = None
            self.idle_model_signature = ""

        assignments = (
            ("global_idle", "p_idle_var", None),
            ("global_kc", "kc_coeff", self._format_optional_model_param),
            ("kc_sigma", "kc_sigma", None),
        )
        for profile_key, variable_name, formatter in assignments:
            if profile_key not in profile:
                continue
            try:
                value = float(profile.get(profile_key))
            except Exception:
                continue
            target = getattr(self, variable_name, None)
            if not np.isfinite(value) or not hasattr(target, "set"):
                continue
            target.set(formatter(value) if callable(formatter) else value)

        ke_value = profile.get("ke_value", profile.get("global_ke"))
        try:
            ke_value = float(ke_value)
        except Exception:
            ke_value = float("nan")
        if np.isfinite(ke_value):
            target = getattr(self, "ke_coeff", None)
            if hasattr(target, "set"):
                target.set(self._format_optional_model_param(ke_value))
        return True

    def _build_profile_prediction_point_map(self, profile):
        if not self.data or not isinstance(profile, dict):
            return {}, {}

        self._ensure_process_point_metadata()
        line_kc_map = self._normalize_profile_line_kc_map(profile)
        point_kc_map = self._normalize_profile_point_kc_map(profile)

        global_kc = self._resolve_profile_global_kc(profile, default=self.get_kc_value())
        for row_index, row in enumerate(self.data):
            try:
                line_no = int(row.get("line_no_aligned", row_index))
            except Exception:
                line_no = int(row_index)
            try:
                point_index = int(row.get("process_point_index", 0))
            except Exception:
                point_index = 0
            point_key = (line_no, point_index)
            if point_key not in point_kc_map and line_no not in line_kc_map:
                point_kc_map[point_key] = max(float(global_kc), 0.0)
        return point_kc_map, line_kc_map

    def _apply_profile_prediction_to_current_data(
        self,
        profile,
        *,
        origin,
        profile_path="",
    ):
        if not self.data or not isinstance(profile, dict):
            return False

        normalized_profile = self._normalize_loaded_kc_profile(
            profile,
            source_path=str(profile_path or ""),
        )
        if not isinstance(normalized_profile, dict):
            return False

        self._activate_profile_state(
            normalized_profile,
            origin=origin,
            file_path=str(profile_path or ""),
            normalize=False,
        )
        self._apply_profile_prediction_parameters(
            normalized_profile,
            source_identity=str(profile_path or ""),
        )
        expected_context_signature = self._build_prediction_context_signature(
            prediction_source=origin,
            measurement=getattr(self, "manual_measurement_data", None),
        )
        self._invalidate_segmentation_for_prediction_context_change(
            expected_context_signature,
            reason="profile 或预测模型发生变化",
        )
        if (
            bool(getattr(self, "_current_interval_ready", False))
            and not self._has_authoritative_segmentation_state()
        ):
            self._clear_current_interval_state(keep_profile_lock=False)

        point_kc_map, line_kc_map = self._build_profile_prediction_point_map(
            normalized_profile
        )
        ke_value = self._resolve_profile_ke_value(
            normalized_profile,
            default=self.get_ke_value(),
        )
        applied = self._apply_point_kc_map_to_current_data(
            point_kc_map,
            line_kc_map=line_kc_map,
            ke_value=ke_value,
            clear_interval_ids=not self._has_authoritative_segmentation_state(),
        )
        if applied:
            self._debug_prediction_state_event(
                "apply_profile_prediction_to_current_data",
                kc_map_source=origin,
                historical_intervals_imported=False,
            )
        return bool(applied)

    def _refresh_segmentation_process_prediction(self, prediction_payload=None):
        """按样本预测来源同步过程域 P/P_idle，并返回可校验的内容签名。"""

        measurement = getattr(self, "manual_measurement_data", None)
        payload = prediction_payload if isinstance(prediction_payload, dict) else measurement

        def _fail(reason):
            self._segmentation_process_prediction_context_signature = ""
            self._segmentation_process_prediction_source = ""
            self._segmentation_process_prediction_row_count = 0
            if isinstance(measurement, dict):
                for key in (
                    "segmentation_process_prediction_context_signature",
                    "segmentation_process_prediction_source",
                    "segmentation_process_prediction_row_count",
                ):
                    measurement.pop(key, None)
            self._debug_prediction_state_event(
                "refresh_segmentation_process_prediction_failed",
                reason=str(reason or "unknown"),
            )
            return {
                "success": False,
                "source": "",
                "context_signature": "",
                "row_count": 0,
                "reason": str(reason or "unknown"),
            }

        if not self.data:
            return _fail("当前没有 ProcessInfo 过程点")
        if not isinstance(payload, dict) or not isinstance(measurement, dict):
            return _fail("缺少实际负载预测 payload")

        policy = payload.get("segmentation_prediction_policy")
        policy = dict(policy) if isinstance(policy, dict) else {}
        declared_source = str(
            policy.get("source")
            or payload.get("segmentation_prediction_source")
            or measurement.get("segmentation_prediction_source")
            or ""
        ).strip()
        profile_origin = self._get_profile_origin()
        if not declared_source and profile_origin == "imported_profile":
            active_profile = getattr(self, "imported_kc_profile", None)
            is_independent = bool(
                isinstance(active_profile, dict)
                and self._profile_is_independent_from_current_measurement(
                    active_profile,
                    measurement=measurement,
                )
            )
            declared_source = "independent_profile" if is_independent else "same_measurement_profile"
        elif not declared_source and isinstance(measurement, dict):
            declared_source = "measurement_reverse"

        expected_policy = {
            "independent_profile": (True, False),
            "same_measurement_profile": (False, True),
            "measurement_reverse": (False, True),
        }
        if declared_source not in expected_policy:
            return _fail(f"不支持的六态预测来源: {declared_source or 'empty'}")
        expected_independent, expected_temporary = expected_policy[declared_source]
        declared_independent = policy.get(
            "independent",
            payload.get("segmentation_prediction_independent"),
        )
        declared_temporary = policy.get(
            "temporary_measurement_mode",
            payload.get("segmentation_temporary_measurement_mode"),
        )
        if declared_independent is not None and bool(declared_independent) != expected_independent:
            return _fail("样本预测独立性标记与来源不一致")
        if declared_temporary is not None and bool(declared_temporary) != expected_temporary:
            return _fail("样本预测临时模式标记与来源不一致")

        applied = False
        if declared_source in {"independent_profile", "same_measurement_profile"}:
            source_profile = self._resolve_imported_profile_for_current_context(
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload=True,
            )
            if not isinstance(source_profile, dict):
                source_profile = getattr(self, "imported_kc_profile", None)
            if not isinstance(source_profile, dict):
                return _fail("样本声明 profile 预测，但当前没有可用 imported profile")
            actual_independent = self._profile_is_independent_from_current_measurement(
                source_profile,
                measurement=measurement,
            )
            if bool(actual_independent) != bool(expected_independent):
                return _fail("当前 profile 与实际负载的绑定关系已变化")
            applied = self._apply_saved_kc_profile_to_current_data(source_profile)
        else:
            runtime_profile = self._resolve_runtime_identified_profile_for_current_case(
                measurement=measurement,
                process_path=self._get_primary_input_file_or_empty(),
            )
            if isinstance(runtime_profile, dict):
                applied = self._apply_profile_prediction_to_current_data(
                    runtime_profile,
                    origin="runtime_identified_profile",
                    profile_path="",
                )
            else:
                aligned_lines = np.asarray(measurement.get("line_no_aligned", []), dtype=int)
                point_indices = np.asarray(measurement.get("process_point_index", []), dtype=int)
                kc_points = self._clip_nonnegative_numeric_array(
                    measurement.get("kc_point", measurement.get("sample_kc_values", []))
                )
                kc_valid_mask = np.asarray(
                    measurement.get("kc_valid_mask", measurement.get("sample_kc_valid_mask", [])),
                    dtype=bool,
                )
                if not (
                    aligned_lines.size
                    == point_indices.size
                    == kc_points.size
                    == kc_valid_mask.size
                    and aligned_lines.size > 0
                    and np.any(kc_valid_mask & np.isfinite(kc_points) & (point_indices >= 0))
                ):
                    return _fail("measurement_reverse 缺少有效的点级 Kc")
                expected_context_signature = self._build_prediction_context_signature(
                    prediction_source="no_profile",
                    measurement=measurement,
                )
                self._invalidate_segmentation_for_prediction_context_change(
                    expected_context_signature,
                    reason="实测反向辨识结果发生变化",
                )
                point_kc_map = self._build_full_point_kc_map_from_current_state(
                    allow_profile_fallback=False,
                    prefer_current_state=False,
                    allow_measurement_point_fallback=True,
                )
                applied = self._apply_point_kc_map_to_current_data(
                    point_kc_map,
                    clear_interval_ids=not self._has_authoritative_segmentation_state(),
                )

        if not applied:
            return _fail("无法用当前预测来源刷新过程域 P/P_idle")

        process_rows = []
        for row_index, row in enumerate(self.data):
            if bool(row.get("_is_synthetic_fill", False)):
                continue
            try:
                predicted_load = float(row.get("P"))
                predicted_idle = float(row.get("P_idle"))
                kc_value = float(row.get("K_c", row.get("K")))
                ke_value = float(row.get("K_e"))
                mrr_value = float(row.get("MRR"))
            except (TypeError, ValueError):
                return _fail(f"过程点 {row_index} 的预测字段不可解析")
            if not all(
                np.isfinite(value)
                for value in (predicted_load, predicted_idle, kc_value, ke_value, mrr_value)
            ):
                return _fail(f"过程点 {row_index} 的预测字段包含非有限值")
            try:
                line_no = int(row.get("line_no_aligned", row_index))
            except Exception:
                line_no = int(row_index)
            try:
                point_index = int(row.get("process_point_index", 0))
            except Exception:
                point_index = 0
            process_rows.append(
                (
                    line_no,
                    point_index,
                    predicted_load,
                    predicted_idle,
                    kc_value,
                    ke_value,
                    mrr_value,
                )
            )

        prediction_context = self._build_prediction_context_signature(
            prediction_source=self._get_prediction_source(),
            measurement=measurement,
        )
        sample_prediction_context = str(
            payload.get("segmentation_sample_prediction_context_signature")
            or measurement.get("segmentation_sample_prediction_context_signature")
            or ""
        )
        if sample_prediction_context and sample_prediction_context != prediction_context:
            return _fail("样本预测与过程预测不属于同一模型上下文")
        context_signature = self._build_stable_prediction_digest(
            {
                "source": declared_source,
                "independent": expected_independent,
                "temporary_measurement_mode": expected_temporary,
                "prediction_context": prediction_context,
                "process_rows": process_rows,
            }
        )
        self._segmentation_process_prediction_context_signature = context_signature
        self._segmentation_process_prediction_source = declared_source
        self._segmentation_process_prediction_row_count = len(process_rows)
        measurement["segmentation_process_prediction_context_signature"] = context_signature
        measurement["segmentation_process_prediction_source"] = declared_source
        measurement["segmentation_process_prediction_row_count"] = len(process_rows)
        self._debug_prediction_state_event(
            "refresh_segmentation_process_prediction",
            segmentation_prediction_source=declared_source,
            segmentation_prediction_independent=expected_independent,
            segmentation_temporary_measurement_mode=expected_temporary,
            process_prediction_context_signature=context_signature,
            process_row_count=len(process_rows),
        )
        return {
            "success": True,
            "source": declared_source,
            "independent": expected_independent,
            "temporary_measurement_mode": expected_temporary,
            "context_signature": context_signature,
            "prediction_context": prediction_context,
            "row_count": len(process_rows),
            "reason": "",
        }

    def _refresh_current_process_prediction_from_runtime(self, allow_profile_fallback=True, prefer_current_state=True):
        if not self.data:
            return False

        prediction_source, source_profile = ("no_profile", None)
        if allow_profile_fallback:
            prediction_source, source_profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload(),
            )
        line_kc_map = (
            self._normalize_profile_line_kc_map(source_profile)
            if allow_profile_fallback and isinstance(source_profile, dict)
            else {}
        )
        point_kc_map = self._build_full_point_kc_map_from_current_state(
            allow_profile_fallback=allow_profile_fallback,
            prefer_current_state=prefer_current_state,
            allow_measurement_point_fallback=False,
        )
        if prediction_source == "imported_profile" and isinstance(source_profile, dict):
            interval_records = self._extract_profile_interval_records(source_profile)
            if interval_records and hasattr(self, "_build_point_kc_map_from_interval_records"):
                point_kc_map = self._build_point_kc_map_from_interval_records(
                    interval_records,
                    base_point_kc_map=point_kc_map,
                )
        if not point_kc_map and not line_kc_map:
            return False
        self._debug_prediction_state_event(
            "refresh_process_prediction_from_runtime",
            kc_map_source=prediction_source if prediction_source != "no_profile" else "current_rows",
        )
        return self._apply_point_kc_map_to_current_data(
            point_kc_map,
            line_kc_map=line_kc_map,
            clear_interval_ids=not bool(getattr(self, "_profile_intervals_locked", False)),
        )

    def _apply_interval_kc_records_to_current_data(self, interval_records, ke_value=None):
        if not self.data or not isinstance(interval_records, list):
            return False

        resolved_ke_value = (
            self._get_effective_ke_value_from_profile(default=self.get_ke_value())
            if ke_value is None
            else max(float(ke_value), 0.0)
        )
        spindle_speeds = []
        for idx, row in enumerate(self.data):
            try:
                spindle_speeds.append(float(row.get("S", self.current_program_speed.get() if hasattr(self, "current_program_speed") else 0.0) or 0.0))
            except Exception:
                spindle_speeds.append(0.0)
        speed_arr = np.asarray(spindle_speeds, dtype=float)

        for idx, interval in enumerate(self._get_steady_interval_records(interval_records), 1):
            try:
                start_idx = int(interval.get("start_idx"))
                end_idx = int(interval.get("end_idx"))
            except Exception:
                start_idx = None
                end_idx = None
            if start_idx is None or end_idx is None:
                continue
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            if start_idx < 0 or start_idx >= len(self.data):
                continue
            end_idx = min(end_idx, len(self.data) - 1)
            interval_id = str(interval.get("zone_id") or interval.get("interval_id") or f"Z{idx:03d}")
            is_idle_interval = bool(interval.get("is_idle_interval")) or str(interval.get("kc_source", "")).strip().lower() == "idle"
            try:
                kc_hat = float(interval.get("K_c_hat"))
            except Exception:
                kc_hat = float("nan")
            if not np.isfinite(kc_hat) and not is_idle_interval:
                interval_kc_values = []
                for row_idx in range(int(start_idx), int(end_idx) + 1):
                    try:
                        row_kc = float(self.data[row_idx].get("K_c", self.data[row_idx].get("K", float("nan"))))
                    except Exception:
                        row_kc = float("nan")
                    if np.isfinite(row_kc):
                        interval_kc_values.append(max(float(row_kc), 0.0))
                kc_hat, _, _ = self._summarize_interval_kc_statistics(interval_kc_values)
            if not is_idle_interval and not np.isfinite(kc_hat):
                continue
            if np.isfinite(kc_hat):
                kc_hat = max(float(kc_hat), 0.0)
            for row_idx in range(int(start_idx), int(end_idx) + 1):
                row = self.data[row_idx]
                idle_power = float(row.get("P_idle", 0.0) or 0.0)
                ap_val = float(row.get("ap", 0.0) or 0.0)
                mrr_val = float(row.get("MRR", 0.0) or 0.0)
                spindle_speed = float(speed_arr[row_idx]) if row_idx < len(speed_arr) else 0.0
                if is_idle_interval:
                    total_power = max(idle_power, 0.0)
                    row["K"] = 0.0
                    row["K_c"] = 0.0
                    row["P_edge"] = 0.0
                else:
                    total_power = max(idle_power + kc_hat * mrr_val + resolved_ke_value * ap_val, 0.0)
                    row["K"] = float(kc_hat)
                    row["K_c"] = float(kc_hat)
                    row["P_edge"] = float(resolved_ke_value * ap_val)
                row["K_e"] = float(resolved_ke_value)
                row["P_idle"] = float(idle_power)
                row["P"] = float(total_power)
                row["steady_interval_id"] = interval_id
                try:
                    angular_velocity = 2 * math.pi * spindle_speed / 60.0
                    row["T"] = float(total_power / angular_velocity) if angular_velocity > 1e-9 else 0.0
                except Exception:
                    row["T"] = 0.0
        self._invalidate_process_alignment_caches(reason="apply_interval_kc_records")
        self._refresh_authoritative_segmentation_interval_descriptors()
        return True

    def _build_compact_runtime_interval_record(
        self,
        start_idx,
        end_idx,
        *,
        zone_id,
        is_idle_interval=False,
        template=None,
        pit_metadata=None,
    ):
        if not self.data:
            return None
        try:
            start_idx = int(start_idx)
            end_idx = int(end_idx)
        except Exception:
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= len(self.data):
            return None

        formatter = getattr(self, "format_line_point", None)
        start_row = self.data[start_idx]
        end_row = self.data[end_idx]
        start_line = self._get_process_row_sample_line(start_row, fallback=start_idx)
        end_line = self._get_process_row_sample_line(end_row, fallback=end_idx)
        try:
            start_point_idx = int(start_row.get("process_point_index", 0))
        except Exception:
            start_point_idx = 0
        try:
            end_point_idx = int(end_row.get("process_point_index", 0))
        except Exception:
            end_point_idx = 0
        if callable(formatter):
            process_start_label = formatter(start_line, start_point_idx)
            process_end_label = formatter(end_line, end_point_idx)
        else:
            process_start_label = f"{int(start_line)}.{int(start_point_idx) + 1}"
            process_end_label = f"{int(end_line)}.{int(end_point_idx) + 1}"

        current = dict(template or {})
        current.update(
            {
                "zone_id": str(zone_id or current.get("zone_id") or ""),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "start_line": int(start_line),
                "end_line": int(end_line),
                "process_start_label": str(process_start_label),
                "process_end_label": str(process_end_label),
                "segment_type": "steady",
                "steady_subtype": "idle" if is_idle_interval else "cutting",
                "is_idle_interval": bool(is_idle_interval),
            }
        )

        try:
            start_s = float(start_row.get("path_start"))
        except Exception:
            start_s = float("nan")
        try:
            end_s = float(end_row.get("path_end"))
        except Exception:
            end_s = float("nan")
        if np.isfinite(start_s):
            current["start_s"] = float(start_s)
        if np.isfinite(end_s):
            current["end_s"] = float(end_s)

        process_bounds = {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_line": int(start_line),
            "end_line": int(end_line),
        }
        process_x_bounds = self._resolve_interval_process_x_bounds(current, process_bounds=process_bounds)
        if process_x_bounds:
            current.update(process_x_bounds)
        sample_bounds = self._resolve_interval_sample_bounds(current)
        if sample_bounds:
            current.update(sample_bounds)
        current["sample_count"] = int(
            current.get("sample_count", end_idx - start_idx + 1) or (end_idx - start_idx + 1)
        )

        geometry = self.summarize_process_interval(start_idx, end_idx) if hasattr(self, "summarize_process_interval") else {}
        rows = self.data[start_idx:end_idx + 1]
        row_pred = []
        row_idle = []
        row_kc = []
        for row in rows:
            try:
                pred_value = float(row.get("P", float("nan")))
            except Exception:
                pred_value = float("nan")
            if np.isfinite(pred_value):
                row_pred.append(float(pred_value))
            try:
                idle_value = float(row.get("P_idle", float("nan")))
            except Exception:
                idle_value = float("nan")
            if np.isfinite(idle_value):
                row_idle.append(float(idle_value))
            try:
                kc_value = float(row.get("K_c", row.get("K", float("nan"))))
            except Exception:
                kc_value = float("nan")
            if np.isfinite(kc_value):
                row_kc.append(max(float(kc_value), 0.0))

        current["a_p"] = float(geometry.get("a_p", current.get("a_p", 0.0)) or 0.0)
        current["a_e"] = float(geometry.get("a_e", current.get("a_e", 0.0)) or 0.0)
        current["F_plan"] = float(geometry.get("F_plan", current.get("F_plan", 0.0)) or 0.0)
        current["p_idle"] = (
            float(np.mean(np.asarray(row_idle, dtype=float)))
            if row_idle else float(geometry.get("p_idle", current.get("p_idle", 0.0)) or 0.0)
        )
        current["p_pred"] = (
            float(np.mean(np.asarray(row_pred, dtype=float)))
            if row_pred else float(current.get("p_pred", 0.0) or 0.0)
        )

        resolved_metadata = pit_metadata or (
            self.get_current_pit_metadata() if hasattr(self, "get_current_pit_metadata") else {}
        )
        current["tool_diameter"] = resolved_metadata.get("tool_diameter")
        current["tool_radius"] = resolved_metadata.get("tool_radius")
        current["tool_material"] = resolved_metadata.get("tool_material")
        current["blank_material"] = resolved_metadata.get("blank_material")

        measurement_summary = None
        start_x = current.get("display_start_x")
        end_x = current.get("display_end_x")
        if hasattr(self, "summarize_measurement_interval") and np.isfinite(start_x) and np.isfinite(end_x):
            measurement_summary = self.summarize_measurement_interval(start_x, end_x)
        if measurement_summary:
            current["p_meas"] = float(measurement_summary.get("p_meas", float("nan")))
            current["actual_load_std"] = float(measurement_summary.get("actual_load_std", 0.0))
            current["actual_load_diff_std"] = float(measurement_summary.get("actual_load_diff_std", 0.0))
            current["valid_kc_count"] = int(measurement_summary.get("valid_kc_count", 0) or 0)
            current["gated_out_count"] = int(measurement_summary.get("gated_out_count", 0) or 0)
            current["sigma_idle"] = float(measurement_summary.get("sigma_idle", 0.0) or 0.0)
            current["delta_mrr"] = float(measurement_summary.get("delta_mrr", 0.0) or 0.0)
            current["steady_pass"] = bool(measurement_summary.get("steady_pass", current.get("steady_pass", True)))
            summary_kc = float(measurement_summary.get("kc_hat", float("nan")))
            summary_sigma = float(measurement_summary.get("sigma_kc", float("nan")))
        else:
            summary_kc = float("nan")
            summary_sigma = float("nan")

        if is_idle_interval:
            current["K_c_hat"] = 0.0
            current["sigma_Kc"] = 0.0
            current["K_c_UCB"] = 0.0
            current["kc_source"] = "idle"
        else:
            if np.isfinite(summary_kc):
                current["K_c_hat"] = max(float(summary_kc), 0.0)
                current["sigma_Kc"] = max(float(summary_sigma) if np.isfinite(summary_sigma) else 0.0, 0.0)
                current["kc_source"] = "measurement_summary"
            elif row_kc:
                row_kc_arr = np.asarray(row_kc, dtype=float)
                current["K_c_hat"] = max(float(np.median(row_kc_arr)), 0.0)
                sigma_kc = self._robust_sigma(row_kc_arr) if row_kc_arr.size > 1 else 0.0
                current["sigma_Kc"] = max(float(sigma_kc) if np.isfinite(sigma_kc) else 0.0, 0.0)
                current["kc_source"] = "interval_median"
            else:
                current["K_c_hat"] = float("nan")
                current["sigma_Kc"] = float("nan")
                current["kc_source"] = ""
        return current

    def _should_merge_compact_runtime_intervals(
        self,
        left_record,
        right_record,
        *,
        max_gap_rows=4,
        kc_abs_tol=0.35,
        kc_ratio_tol=0.18,
    ):
        if not isinstance(left_record, dict) or not isinstance(right_record, dict):
            return False
        if bool(left_record.get("is_idle_interval")) or bool(right_record.get("is_idle_interval")):
            return False
        try:
            left_end = int(left_record.get("end_idx"))
            right_start = int(right_record.get("start_idx"))
        except Exception:
            return False
        if right_start <= left_end:
            gap_rows = 0
        else:
            gap_rows = int(right_start - left_end - 1)
        if gap_rows > int(max_gap_rows):
            return False
        try:
            left_kc = float(left_record.get("K_c_hat"))
            right_kc = float(right_record.get("K_c_hat"))
        except Exception:
            return False
        if not (np.isfinite(left_kc) and np.isfinite(right_kc)):
            return False
        if left_kc <= 1e-9 or right_kc <= 1e-9:
            return False
        return abs(left_kc - right_kc) <= max(
            float(kc_abs_tol),
            float(kc_ratio_tol) * max(abs(left_kc), abs(right_kc), 1.0),
        )

    def _build_compact_runtime_intervals_from_segments(
        self,
        segment_records,
        *,
        include_idle=False,
        min_process_rows=1,
        min_sample_count=20,
    ):
        if not self.data or not isinstance(segment_records, list):
            return []

        pit_metadata = self.get_current_pit_metadata() if hasattr(self, "get_current_pit_metadata") else {
            "tool_diameter": None,
            "tool_radius": None,
            "tool_material": "",
            "blank_material": "",
        }
        compact_records = []
        zone_index = 1

        for segment in segment_records:
            if not isinstance(segment, dict):
                continue
            state_code = self._resolve_smif_state_code(segment)
            if int(state_code) == 0:
                continue
            is_idle_interval = int(state_code) == 1 or bool(segment.get("is_idle_interval"))
            if is_idle_interval and not include_idle:
                continue

            process_bounds = self._resolve_interval_process_bounds(segment)
            if not process_bounds:
                continue
            start_idx = int(process_bounds["start_idx"])
            end_idx = int(process_bounds["end_idx"])
            process_row_count = int(end_idx - start_idx + 1)
            if process_row_count <= 0:
                continue

            current = self._extract_interval_template_record(segment) or dict(segment)
            sample_bounds = self._resolve_interval_sample_bounds(current)
            if sample_bounds:
                current.update(sample_bounds)
            sample_count = int(current.get("sample_count", process_row_count) or process_row_count)
            if not is_idle_interval and (
                process_row_count < int(min_process_rows)
                or sample_count < int(min_sample_count)
            ):
                continue

            record = self._build_compact_runtime_interval_record(
                start_idx,
                end_idx,
                zone_id=str(current.get("zone_id") or current.get("interval_id") or f"Z{zone_index:03d}"),
                is_idle_interval=is_idle_interval,
                template=current,
                pit_metadata=pit_metadata,
            )
            if record is None:
                continue
            compact_records.append(record)
            zone_index += 1

        if not compact_records:
            return []

        normalized_records = []
        for record in compact_records:
            if not isinstance(record, dict):
                continue
            if not bool(record.get("is_idle_interval")):
                try:
                    record_kc = float(record.get("K_c_hat"))
                except Exception:
                    record_kc = float("nan")
                if np.isfinite(record_kc) and record_kc <= 1e-9:
                    continue
            if normalized_records and self._should_merge_compact_runtime_intervals(normalized_records[-1], record):
                merged_record = self._build_compact_runtime_interval_record(
                    int(normalized_records[-1].get("start_idx")),
                    int(record.get("end_idx")),
                    zone_id=str(normalized_records[-1].get("zone_id") or record.get("zone_id") or f"Z{len(normalized_records):03d}"),
                    is_idle_interval=False,
                    template=normalized_records[-1],
                    pit_metadata=pit_metadata,
                )
                if merged_record is not None:
                    normalized_records[-1] = merged_record
                    continue
            normalized_records.append(record)
        if not normalized_records:
            return []
        return self.finalize_interval_kc(normalized_records)

    def _apply_saved_kc_profile_to_current_data(self, profile=None):
        if not self.data:
            return False

        source_profile = profile if isinstance(profile, dict) else self._resolve_imported_profile_for_current_context(
            process_path=self._get_primary_input_file_or_empty(),
            allow_autoload=True,
        )
        profile = self._normalize_loaded_kc_profile(source_profile, source_path=str(getattr(self, "active_kc_profile_path", "") or ""))
        if not isinstance(profile, dict):
            if not self._has_authoritative_segmentation_state():
                self._clear_current_interval_state(keep_profile_lock=False)
            self.refresh_pit_button_state()
            return False

        profile_is_independent = self._profile_is_independent_from_current_measurement(profile)
        if not profile_is_independent:
            self._debug_prediction_state_event(
                "allow_measurement_bound_profile_for_temporary_segmentation",
                reason="profile_training_data_matches_current_measurement",
                kc_map_source="same_measurement_profile",
                segmentation_prediction_independent=False,
                segmentation_temporary_measurement_mode=True,
            )

        profile_path = str(
            getattr(self, "imported_kc_profile_path", "") or getattr(self, "active_kc_profile_path", "") or ""
        ).strip()
        applied = self._apply_profile_prediction_to_current_data(
            profile,
            origin="imported_profile",
            profile_path=profile_path,
        )
        self.refresh_pit_button_state()
        return bool(applied)

    def _refresh_preview_with_saved_kc_profile(self, refresh_measurement=True):
        self._apply_saved_kc_profile_to_current_data(getattr(self, "imported_kc_profile", None))
        resolved_interval_policy = "use_active_profile"
        resolved_refresh_prediction = True
        if refresh_measurement and getattr(self, "manual_measurement_data", None):
            self._refresh_manual_measurement_prediction(
                allow_saved_sample_profile=False,
                allow_measurement_resolve=False,
                display_mode="forward",
            )
        if self.data:
            self.generate_plots(
                save=False,
                silent=True,
                interval_policy=resolved_interval_policy,
                persist_profile=False,
                refresh_prediction=resolved_refresh_prediction,
            )
        if hasattr(self, "_schedule_smif_refresh"):
            try:
                self._schedule_smif_refresh(delay_ms=0)
            except Exception:
                pass
        elif hasattr(self, "refresh_smif_view"):
            try:
                self.refresh_smif_view()
            except Exception:
                pass

    def _sanitize_kc_profile_stem(self, text):
        stem = re.sub(r'[\\\\/:*?\"<>|]+', "_", str(text or "").strip())
        stem = re.sub(r"\s+", "_", stem).strip("._")
        return stem or "gcode_case"

    def _build_default_kc_profile_filename(self):
        case_stem = ""
        for candidate in (
            getattr(self, "gcode_nc_path_var", None).get() if hasattr(getattr(self, "gcode_nc_path_var", None), "get") else "",
            self.get_primary_input_file(),
        ):
            if candidate:
                case_stem = os.path.splitext(os.path.basename(str(candidate)))[0]
                break
        case_stem = self._sanitize_kc_profile_stem(case_stem)
        default_name = f"{case_stem}.kcke"
        if not os.path.isdir(self.kc_profile_dir):
            return default_name

        default_path = os.path.join(self.kc_profile_dir, default_name)
        if not os.path.exists(default_path):
            return default_name

        pattern = re.compile(rf"^{re.escape(case_stem)}_(\d+)\.kcke(?:\.json)?$", re.IGNORECASE)
        next_index = 1
        for entry in os.listdir(self.kc_profile_dir):
            match = pattern.match(entry)
            if not match:
                continue
            try:
                next_index = max(next_index, int(match.group(1)) + 1)
            except Exception:
                continue
        return f"{case_stem}_{next_index:03d}.kcke"

    def _wrap_kc_profile_file_payload(self, profile, active_profile_path=""):
        if not isinstance(profile, dict):
            return None
        profile = self._normalize_loaded_kc_profile(
            self._strip_measurement_bound_fields_from_profile(
                profile,
                keep_sample_kc_profile=True,
            ),
            source_path=str(active_profile_path or ""),
        )
        if not isinstance(profile, dict):
            return None

        case_name = self._sanitize_kc_profile_stem(
            os.path.splitext(os.path.basename(self.gcode_nc_path_var.get().strip()))[0]
            if self.gcode_nc_path_var.get().strip()
            else os.path.splitext(os.path.basename(self.get_primary_input_file() or "gcode_case"))[0]
        )
        top_level_template = {
            "template_schema_version": int(profile.get("template_schema_version", 3) or 3),
            "template_context": dict(profile.get("template_context") or {}),
            "global_kc": profile.get("global_kc"),
            "global_ke": profile.get("global_ke"),
            "ke_value": profile.get("ke_value"),
            "global_idle": profile.get("global_idle"),
            "idle_power_model": dict(profile.get("idle_power_model") or {}),
            "point_kc_map": dict(profile.get("point_kc_map") or {}),
            "pit_records": self._serialize_record_list(profile.get("pit_records") or []),
            "segment_records": self._serialize_record_list(profile.get("segment_records") or []),
            "sample_kc_profile": dict(profile.get("sample_kc_profile") or {}) if isinstance(profile.get("sample_kc_profile"), dict) else None,
            "interval_templates": self._serialize_record_list(profile.get("interval_templates") or []),
            "interval_count": int(profile.get("interval_count", len(profile.get("pit_records") or [])) or 0),
            "segment_count": int(profile.get("segment_count", len(profile.get("segment_records") or [])) or 0),
            "point_kc_count": int(profile.get("point_kc_count", len(profile.get("point_kc_map") or {})) or 0),
        }
        saved_profile = dict(profile)
        for stale_key in (
            "point_actual_feed_map",
            "point_actual_mrr_map",
            "sample_forward_semantics",
            "feed_source_semantics",
        ):
            saved_profile.pop(stale_key, None)
        return {
            "format_version": 6,
            "profile_type": "kcke_case_profile",
            "profile_name": case_name,
            "saved_at": datetime.now().isoformat(),
            "active_profile_path": str(active_profile_path or ""),
            **top_level_template,
            "profile": {key: value for key, value in saved_profile.items()},
        }

    def _build_kc_profile_file_payload(self):
        profile = getattr(self, "active_kc_profile", None)
        should_sync_current = bool(getattr(self, "_current_interval_ready", False))
        snapshot_source = "manual_save"
        if self._get_profile_origin() == "runtime_identified_profile":
            snapshot_source = str((profile or {}).get("source") or "measurement")
        if should_sync_current or not self._profile_has_saved_payload(profile):
            profile = self._build_current_kc_profile_snapshot(source=snapshot_source)
        if not isinstance(profile, dict):
            profile = getattr(self, "active_kc_profile", None)
        if not isinstance(profile, dict):
            return None

        profile = dict(profile)
        if self._get_profile_origin() == "runtime_identified_profile":
            profile["source"] = str(profile.get("source") or snapshot_source or "measurement")
        profile["global_idle"] = float(self.p_idle_var.get() or profile.get("global_idle", 0.0) or 0.0)
        profile["global_kc"] = float(self.get_kc_value(profile.get("global_kc", 0.0)))
        profile["kc_sigma"] = float(self.kc_sigma.get() or profile.get("kc_sigma", 0.0) or 0.0)
        profile["global_ke"] = float(self._get_effective_ke_value_from_profile(profile, default=profile.get("global_ke", 0.0) or 0.0))
        profile["ke_value"] = float(profile["global_ke"])
        profile["idle_power_model"] = dict(getattr(self, "idle_power_model", {}) or {})
        profile["idle_model_signature"] = str(getattr(self, "idle_model_signature", "") or "")
        profile["updated_at"] = datetime.now().isoformat()
        profile["template_schema_version"] = 3
        profile["template_context"] = self._build_profile_template_context(process_path=profile.get("process_path"), gcode_path=profile.get("gcode_path"))
        for stale_key in (
            "point_actual_feed_map",
            "point_actual_mrr_map",
            "sample_forward_semantics",
            "feed_source_semantics",
        ):
            profile.pop(stale_key, None)
        normalized_profile = self._normalize_loaded_kc_profile(
            self._strip_measurement_bound_fields_from_profile(
                profile,
                keep_sample_kc_profile=True,
            ),
            source_path=str(getattr(self, "active_kc_profile_path", "") or ""),
        )
        if self._get_profile_origin() == "imported_profile":
            self.imported_kc_profile = dict(normalized_profile) if isinstance(normalized_profile, dict) else None
        elif self._get_profile_origin() == "runtime_identified_profile":
            self.runtime_identified_kc_profile = dict(normalized_profile) if isinstance(normalized_profile, dict) else None
        self.active_kc_profile = normalized_profile
        return self._wrap_kc_profile_file_payload(
            self.active_kc_profile,
            active_profile_path=str(getattr(self, "active_kc_profile_path", "") or ""),
        )

    def _apply_loaded_kc_profile(self, profile, file_path="", refresh_preview=True):
        if not isinstance(profile, dict):
            raise ValueError("配置文件内容无效")

        preserve_segmentation = self._has_authoritative_segmentation_state()
        normalized_profile = self._normalize_loaded_kc_profile(profile, source_path=file_path)
        self._clear_runtime_identified_profile_state(clear_active=False, reason="load_imported_profile")
        self._invalidate_measurement_runtime_state(
            keep_profile_lock=False,
            clear_interval_state=not preserve_segmentation,
        )
        self._activate_profile_state(
            normalized_profile,
            origin="imported_profile",
            file_path=str(file_path or ""),
            normalize=False,
        )
        self._set_profile_import_skip_state(skipped=False)
        self._apply_profile_prediction_parameters(
            normalized_profile,
            source_identity=str(file_path or ""),
        )

        if self.gcode_nc_path_var.get().strip() and os.path.exists(self.gcode_nc_path_var.get().strip()):
            self._refresh_current_program_idle_power_from_gcode()
        else:
            self.current_program_idle_power.set(float(self.p_idle_var.get() or 0.0))
            self._update_program_idle_summary()

        expected_context_signature = self._build_prediction_context_signature(
            prediction_source="imported_profile",
            measurement=getattr(self, "manual_measurement_data", None),
        )
        self._invalidate_segmentation_for_prediction_context_change(
            expected_context_signature,
            reason="导入的 profile 或预测模型发生变化",
        )
        if (
            bool(getattr(self, "_current_interval_ready", False))
            and not self._has_authoritative_segmentation_state()
        ):
            self._clear_current_interval_state(keep_profile_lock=False)
        self._debug_interval_state_event(
            "activate_profile_parameters_only",
            source="imported_profile",
            profile_path=os.path.basename(file_path) if file_path else "memory",
            historical_intervals_imported=False,
        )
        self.refresh_pit_button_state()

        profile_name = os.path.basename(file_path) if file_path else "当前内存配置"
        self.kc_profile_status_var.set(f"案例配置: {profile_name}")
        self.step_feed_status_var.set(f"已加载案例配置: {profile_name}")
        self.refresh_mechanism_status_summary()
        self._debug_prediction_state_event(
            "activate_imported_profile",
            measurement_case_signature=self._get_current_measurement_case_signature() or "none",
            reverse_solve=False,
            kc_map_source="imported_profile",
        )
        if file_path:
            self._register_gcode_profile_binding(normalized_profile.get("gcode_path") or self.gcode_nc_path_var.get(), file_path, persist=False)
            profile_key = self._build_process_kc_profile_key(normalized_profile.get("process_path"))
            if profile_key:
                self.saved_kc_profiles[profile_key] = dict(normalized_profile)
                self._register_saved_kc_profile_index(profile_key, file_path, normalized_profile, persist=False)
        self._persist_app_config()
        if refresh_preview:
            self._refresh_preview_with_saved_kc_profile()
        if hasattr(self, "_refresh_ideal_tree"):
            self._refresh_ideal_tree()

    def _load_kc_case_profile_from_path(self, file_path, refresh_preview=True, update_status=True):
        if not file_path:
            return False
        profile = self._load_kc_profile_payload_from_path(file_path)
        if not isinstance(profile, dict):
            raise ValueError("未识别到有效的 profile 节点")
        self._apply_loaded_kc_profile(profile, file_path=file_path, refresh_preview=refresh_preview)
        if update_status:
            self.set_status(f"已加载案例配置: {os.path.basename(file_path)}", 4000)
        return True

    def _show_kc_profile_selection_dialog(self, gcode_path, candidate_paths):
        options = [str(path).strip() for path in candidate_paths if str(path).strip()]
        if not options:
            return ""

        dialog = tk.Toplevel(self.root)
        dialog.title("选择参数配置")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        result = {"path": ""}

        title_text = f"检测到与 {os.path.basename(str(gcode_path or '').strip())} 匹配的参数配置文件："
        ttk.Label(dialog, text=title_text, font=UI_FONT_NORMAL).pack(anchor="w", padx=12, pady=(12, 6))
        ttk.Label(dialog, text="请选择要导入的配置；若不导入则需要重新辨识。", font=UI_FONT_SMALL).pack(anchor="w", padx=12, pady=(0, 8))

        listbox = tk.Listbox(dialog, width=72, height=min(8, max(3, len(options))), exportselection=False)
        profile_root = str(getattr(self, "kc_profile_dir", "") or "").strip()
        for idx, profile_path in enumerate(options):
            display_text = os.path.basename(profile_path)
            try:
                if profile_root:
                    relative_path = os.path.relpath(profile_path, profile_root)
                    if not relative_path.startswith(".."):
                        display_text = relative_path
            except Exception:
                pass
            listbox.insert(idx, display_text)
        listbox.selection_set(0)
        listbox.pack(fill=tk.BOTH, expand=True, padx=12)

        def _load_selected():
            selection = listbox.curselection()
            if selection:
                result["path"] = options[int(selection[0])]
            dialog.destroy()

        def _skip():
            result["path"] = ""
            dialog.destroy()

        btn_frame = ttk.Frame(dialog, padding=(12, 10, 12, 12))
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="导入选中配置", command=_load_selected, width=14).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="不导入", command=_skip, width=10).pack(side=tk.RIGHT)

        dialog.bind("<Double-Button-1>", lambda event: _load_selected())
        center_dialog_on_parent(dialog, self.root)
        self.root.wait_window(dialog)
        return str(result.get("path") or "")

    def _handle_kc_profile_after_gcode_import(self, gcode_path=None):
        gcode_path = str(gcode_path or self.gcode_nc_path_var.get() or "").strip()
        if not gcode_path:
            return False

        if self._active_kc_profile_matches_gcode(gcode_path):
            self._set_profile_import_skip_state(gcode_path, skipped=False)
            active_profile_path = str(getattr(self, "active_kc_profile_path", "") or "").strip()
            if active_profile_path and os.path.exists(active_profile_path):
                self._register_gcode_profile_binding(gcode_path, active_profile_path, persist=False)
                profile_name = os.path.basename(active_profile_path)
            else:
                profile_name = "当前内存配置"
            if hasattr(self, "smif_scope_var"):
                try:
                    self.smif_scope_var.set(
                        "steady" if self._profile_contains_steady_interval_records(getattr(self, "active_kc_profile", None)) else "all"
                    )
                except Exception:
                    pass
            self.refresh_mechanism_status_summary()
            self._persist_app_config()
            self._refresh_preview_with_saved_kc_profile()
            self.set_status(f"已沿用参数配置: {profile_name}", 5000)
            return True

        candidates = self._collect_kc_profile_file_candidates_for_gcode(gcode_path)
        if not candidates:
            self._set_profile_import_skip_state(gcode_path, skipped=True)
            self.clear_kc_ke_state(persist=False, status_text="未绑定参数配置，请重新辨识")
            self.kc_profile_status_var.set("案例配置: 未找到NC绑定配置")
            self.refresh_mechanism_status_summary()
            self._persist_app_config()
            self.set_status("未找到对应参数配置，请点击“重新辨识”", 5000)
            return False

        if len(candidates) == 1:
            selected_path = str(candidates[0])
        else:
            selected_path = self._show_kc_profile_selection_dialog(gcode_path, candidates)
        if not selected_path:
            self._set_profile_import_skip_state(gcode_path, skipped=True)
            self.clear_kc_ke_state(persist=False, status_text="未导入参数配置，请重新辨识")
            self.kc_profile_status_var.set("案例配置: 未导入")
            self.refresh_mechanism_status_summary()
            self._persist_app_config()
            self.set_status("已跳过导入参数配置，请点击“重新辨识”", 5000)
            return False

        self._set_profile_import_skip_state(gcode_path, skipped=False)
        self._load_kc_case_profile_from_path(selected_path, refresh_preview=True, update_status=False)
        self.set_status(f"已导入参数配置: {os.path.basename(selected_path)}", 5000)
        return True

    def load_kc_case_profile_file(self):
        file_path = filedialog.askopenfilename(
            title="选择 Kc/Ke 案例配置",
            initialdir=self.kc_profile_dir,
            filetypes=(("KcKe配置", "*.kcke *.kcke.json"), ("JSON文件", "*.json"), ("所有文件", "*.*")),
        )
        if not file_path:
            return
        try:
            self._load_kc_case_profile_from_path(file_path, refresh_preview=True, update_status=True)
        except Exception as exc:
            messagebox.showerror("加载失败", f"加载案例配置失败:\n{str(exc)}")

    def save_active_kc_case_profile(self, save_as=False, target_path=None, auto_default=False):
        if hasattr(self, "_cancel_pending_sample_selection_change"):
            try:
                self._cancel_pending_sample_selection_change()
            except Exception:
                pass
        payload = self._build_kc_profile_file_payload()
        if not payload:
            messagebox.showwarning("无法保存", "当前还没有可保存的 Kc/Ke 案例配置")
            return False

        resolved_target_path = str(target_path or getattr(self, "active_kc_profile_path", "") or "").strip()
        if save_as or not resolved_target_path:
            default_name = self._build_default_kc_profile_filename()
            if auto_default:
                os.makedirs(self.kc_profile_dir, exist_ok=True)
                resolved_target_path = os.path.join(self.kc_profile_dir, default_name)
            else:
                resolved_target_path = filedialog.asksaveasfilename(
                    title="保存 Kc/Ke 案例配置",
                    initialdir=self.kc_profile_dir,
                    initialfile=default_name,
                    defaultextension=".kcke",
                    filetypes=(("KcKe配置", "*.kcke"), ("兼容旧配置", "*.kcke.json"), ("JSON文件", "*.json")),
                )
            if not resolved_target_path:
                return False

        try:
            with open(resolved_target_path, "w", encoding="utf-8") as outfile:
                json.dump(payload, outfile, ensure_ascii=False, indent=2)
            self.active_kc_profile_path = resolved_target_path
            self._register_gcode_profile_binding(self.gcode_nc_path_var.get(), resolved_target_path, persist=False)
            profile_key = self._build_process_kc_profile_key(payload["profile"].get("process_path"))
            if profile_key:
                self.saved_kc_profiles[profile_key] = dict(payload["profile"])
                self._register_saved_kc_profile_index(profile_key, resolved_target_path, payload["profile"], persist=False)
            self.kc_profile_status_var.set(f"案例配置: {os.path.basename(resolved_target_path)}")
            self.refresh_mechanism_status_summary()
            self._persist_app_config()
            if hasattr(self, "refresh_prediction_metrics_summary"):
                self.refresh_prediction_metrics_summary()
            self.set_status(f"案例配置已保存: {os.path.basename(resolved_target_path)}", 4000)
            return True
        except Exception as exc:
            messagebox.showerror("保存失败", f"保存案例配置失败:\n{str(exc)}")
            return False

    def _persist_identified_profile(self, save_strategy="prompt", persist_current=True):
        if persist_current:
            if not self._persist_current_kc_profile(source="measurement"):
                return False
        else:
            active_profile = getattr(self, "active_kc_profile", None)
            if not self._profile_has_saved_payload(active_profile):
                return False
        if save_strategy == "prompt":
            return bool(self._prompt_save_profile_after_identification())
        if save_strategy == "auto_default":
            gcode_path = ""
            gcode_var = getattr(self, "gcode_nc_path_var", None)
            if hasattr(gcode_var, "get"):
                gcode_path = str(gcode_var.get() or "").strip()
            if gcode_path:
                return bool(self.save_active_kc_case_profile(save_as=True, auto_default=True))
            self.set_status("参数辨识已完成，但当前未导入G代码，配置仅保留在本次会话内", 5000)
            return False
        return False

    def _ensure_prediction_model_for_current_process(self, auto_identify_missing=False, save_strategy="prompt"):
        active_profile = getattr(self, "imported_kc_profile", None)
        has_active_profile = (
            isinstance(active_profile, dict)
            and self._profile_has_saved_payload(active_profile)
            and self._profile_matches_current_context(active_profile)
        )
        skip_auto_profile = self._should_skip_auto_profile_for_current_gcode() and not has_active_profile
        profile_applied = False if skip_auto_profile else bool(self._apply_saved_kc_profile_to_current_data())
        if profile_applied:
            if getattr(self, "sample_data_mode", "") == "experiment_measurement" and getattr(self, "manual_measurement_data", None):
                self._refresh_manual_measurement_prediction()
            return "profile"

        runtime_profile = self._resolve_runtime_identified_profile_for_current_case(
            measurement=getattr(self, "manual_measurement_data", None),
            process_path=self._get_primary_input_file_or_empty(),
        )
        if isinstance(runtime_profile, dict):
            self._activate_profile_state(
                runtime_profile,
                origin="runtime_identified_profile",
                file_path="",
                case_signature=self._get_current_measurement_case_signature(),
            )
            self._refresh_current_process_prediction_from_runtime(
                allow_profile_fallback=True,
                prefer_current_state=True,
            )
            if getattr(self, "sample_data_mode", "") == "experiment_measurement" and getattr(self, "manual_measurement_data", None):
                self._refresh_manual_measurement_prediction(
                    allow_saved_sample_profile=False,
                    display_mode=self._get_measurement_display_mode(prediction_source="runtime_identified_profile"),
                )
            return "runtime_profile"

        if (
            auto_identify_missing
            and getattr(self, "sample_data_mode", "") == "experiment_measurement"
            and getattr(self, "manual_measurement_data", None)
        ):
            # 实际负载只用于分析和采样坐标，不得被自动反解为六态分类模型。
            # 缺少独立 profile 时保留实测分析预览，但不写回过程域 Kc/Ke/P_idle。
            self._debug_prediction_state_event(
                "skip_measurement_auto_identification",
                reason="actual_measurement_is_not_a_segmentation_model_source",
            )

        if getattr(self, "sample_data_mode", "") == "experiment_measurement" and getattr(self, "manual_measurement_data", None):
            self._refresh_manual_measurement_prediction()
            return "measurement_only"
        return "none"

    def _resolve_step_feed_geometry(self, df, ap_col=None, ae_col=None):
        ap_val = None
        ae_val = None

        if ap_col:
            ap_series = pd.to_numeric(df[ap_col], errors='coerce').dropna()
            if not ap_series.empty:
                ap_val = float(ap_series.median())

        if ae_col:
            ae_series = pd.to_numeric(df[ae_col], errors='coerce').dropna()
            if not ae_series.empty:
                ae_val = float(ae_series.median())

        if ap_val is None or ae_val is None:
            for row in self.data:
                try:
                    row_ap = float(row.get('ap', 0.0))
                    row_ae = float(row.get('ae', 0.0))
                except Exception:
                    continue
                if row_ap > 0 and row_ae > 0:
                    ap_val = row_ap if ap_val is None else ap_val
                    ae_val = row_ae if ae_val is None else ae_val
                    break

        if ap_val is None or ae_val is None:
            raise ValueError("阶梯进给CSV缺少ap/ae信息，且当前未加载可推断几何参数的工艺信息表")

        return ap_val, ae_val

    def _relative_span(self, values):
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        center = float(np.median(finite))
        low = float(np.percentile(finite, 5))
        high = float(np.percentile(finite, 95))
        scale = max(abs(center), 1.0)
        return abs(high - low) / scale

    def _relative_span_from_bounds(self, min_value, max_value):
        try:
            low = float(min_value)
            high = float(max_value)
        except Exception:
            return float("nan")
        if not np.isfinite(low) or not np.isfinite(high):
            return float("nan")
        center = (low + high) * 0.5
        scale = max(abs(center), 1.0)
        return abs(high - low) / scale

    def _relative_difference(self, left, right):
        try:
            left_val = float(left)
            right_val = float(right)
        except Exception:
            return float("inf")
        scale = max(abs(left_val), abs(right_val), 1.0)
        return abs(left_val - right_val) / scale

    def _robust_sigma(self, values):
        finite = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(dtype=float)
        if finite.size == 0:
            return float("nan")
        if finite.size == 1:
            return 0.0
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 1e-12:
            return float(robust_sigma)
        return float(np.std(finite, ddof=1))

    def _filter_interval_kc_values(self, values, window=5):
        finite = self._clip_nonnegative_numeric_array(values)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return np.asarray([], dtype=float)
        try:
            window_size = int(window)
        except Exception:
            window_size = 5
        if window_size <= 1 or finite.size <= 2:
            return finite
        if window_size % 2 == 0:
            window_size += 1
        max_window = int(finite.size if finite.size % 2 == 1 else finite.size - 1)
        window_size = min(window_size, max_window)
        if window_size <= 1:
            return finite
        filtered = (
            pd.Series(finite, dtype=float)
            .rolling(window=window_size, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )
        filtered = filtered[np.isfinite(filtered)]
        return filtered if filtered.size > 0 else finite

    def _summarize_interval_kc_mode_statistics(self, values, precision=6):
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        valid = arr[np.isfinite(arr) & (arr >= 0.0)]
        if valid.size == 0:
            return float("nan"), float("nan"), valid

        rounded = np.round(valid, int(precision))
        unique_values, counts = np.unique(rounded, return_counts=True)
        max_count = int(np.max(counts))
        candidates = unique_values[counts == max_count]
        center = float(np.median(valid))
        kc_hat = float(candidates[np.argmin(np.abs(candidates - center))])
        sigma_kc = self._robust_sigma(valid) if valid.size > 1 else 0.0
        if not np.isfinite(sigma_kc):
            sigma_kc = 0.0
        return max(float(kc_hat), 0.0), max(float(sigma_kc), 0.0), valid

    def _summarize_interval_kc_statistics(self, values, window=5):
        return self._summarize_interval_kc_mode_statistics(values, precision=6)

    def _format_metric_text(self, value, unit=""):
        try:
            numeric = float(value)
        except Exception:
            return "未计算"
        if not np.isfinite(numeric):
            return "未计算"
        suffix = f" {unit}" if unit else ""
        return f"{numeric:.6f}{suffix}".rstrip("0").rstrip(".")

    def _resolve_fixed_ke_for_identification(self):
        lock_var = getattr(self, "lock_ke_during_identification", None)
        if lock_var is None or not bool(lock_var.get()):
            return None
        return self._parse_optional_float(self.ke_coeff.get())

    def _get_identification_short_label(self):
        return "Kc" if self._resolve_fixed_ke_for_identification() is not None else "Kc/Ke"

    def _get_identification_target_label(self):
        return "K_c" if self._resolve_fixed_ke_for_identification() is not None else "K_c / K_e"

    def _format_identification_mode_text(self, fit_result):
        if fit_result.get("ke_locked", False):
            return "锁定K_e，仅辨识K_c"
        return "联合辨识K_c/K_e"

    def _format_identification_ke_text(self, fit_result):
        if fit_result.get("ke_locked", False):
            return f"全局K_e保持={fit_result['ke_value']:.6f}"
        return f"全局K_e={fit_result['ke_value']:.6f}"

    def _apply_cutting_model_fit_result(self, fit_result):
        fit_result["kc_value"] = max(float(fit_result.get("kc_value", 0.0) or 0.0), 0.0)
        fit_result["ke_value"] = max(float(fit_result.get("ke_value", 0.0) or 0.0), 0.0)
        fit_result["kc_sigma"] = max(float(fit_result.get("kc_sigma", 0.0) or 0.0), 0.0)
        fit_result["kc_ucb"] = max(
            float(fit_result.get("kc_ucb", fit_result["kc_value"]) or fit_result["kc_value"]),
            fit_result["kc_value"],
        )
        self.kc_coeff.set(self._format_optional_model_param(fit_result["kc_value"]))
        self.kc_sigma.set(fit_result["kc_sigma"])
        if not fit_result.get("ke_locked", False) or self._parse_optional_float(self.ke_coeff.get()) is None:
            self.ke_coeff.set(self._format_optional_model_param(fit_result["ke_value"]))

    def _solve_nonnegative_scalar(self, feature_values, target_values):
        feature = np.asarray(feature_values, dtype=float).reshape(-1)
        target = np.asarray(target_values, dtype=float).reshape(-1)
        denom = float(np.dot(feature, feature))
        if denom <= 1e-12:
            return 0.0
        return max(float(np.dot(feature, target) / denom), 0.0)

    def _solve_nonnegative_kc_ke(self, mrr_values, ap_values, y_values):
        mrr_arr = np.asarray(mrr_values, dtype=float).reshape(-1)
        ap_arr = np.asarray(ap_values, dtype=float).reshape(-1)
        y_arr = np.asarray(y_values, dtype=float).reshape(-1)
        design_matrix = np.column_stack([mrr_arr, ap_arr])

        candidates = []
        try:
            coeffs, _, _, _ = np.linalg.lstsq(design_matrix, y_arr, rcond=None)
            coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
            if coeffs.size == 2 and np.all(np.isfinite(coeffs)) and np.all(coeffs >= 0.0):
                candidates.append(coeffs)
        except Exception:
            pass

        candidates.extend([
            np.array([self._solve_nonnegative_scalar(mrr_arr, y_arr), 0.0], dtype=float),
            np.array([0.0, self._solve_nonnegative_scalar(ap_arr, y_arr)], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ])

        best_coeffs = candidates[-1]
        best_rss = float("inf")
        for coeffs in candidates:
            residual = y_arr - design_matrix @ coeffs
            rss = float(np.dot(residual, residual))
            if rss < best_rss:
                best_rss = rss
                best_coeffs = coeffs
        return float(best_coeffs[0]), float(best_coeffs[1])

    def _refresh_lock_ke_check_text(self):
        text_var = getattr(self, "lock_ke_check_text", None)
        lock_var = getattr(self, "lock_ke_during_identification", None)
        if text_var is None or lock_var is None:
            return
        prefix = "√" if bool(lock_var.get()) else "□"
        text_var.set(f"{prefix} 锁定K_e，仅辨识K_c")

    def _refresh_lock_idle_check_text(self):
        text_var = getattr(self, "lock_idle_check_text", None)
        lock_var = getattr(self, "lock_idle_during_identification", None)
        if text_var is None or lock_var is None:
            return
        prefix = "√" if bool(lock_var.get()) else "□"
        text_var.set(f"{prefix} 锁定P_idle，参数辨识不覆盖")

    def _resolve_fixed_global_idle_power(self):
        lock_var = getattr(self, "lock_idle_during_identification", None)
        if lock_var is None or not bool(lock_var.get()):
            return None
        try:
            idle_power = float(self.p_idle_var.get())
        except Exception:
            return None
        if not np.isfinite(idle_power) or idle_power <= 0.0:
            return None
        return float(idle_power)

    def _format_idle_commit_text(self, idle_commit_result):
        committed_idle = float(idle_commit_result.get("committed_idle", 0.0) or 0.0)
        if idle_commit_result.get("idle_locked", False):
            return f"全局P_idle保持={committed_idle:.6f}"
        return f"全局P_idle={committed_idle:.6f}"

    def _prompt_save_profile_after_identification(self):
        if hasattr(self, "_cancel_pending_sample_selection_change"):
            try:
                self._cancel_pending_sample_selection_change()
            except Exception:
                pass
        active_path = str(getattr(self, "active_kc_profile_path", "") or "").strip()
        has_overwrite_target = bool(active_path)
        gcode_path = ""
        gcode_var = getattr(self, "gcode_nc_path_var", None)
        if hasattr(gcode_var, "get"):
            gcode_path = str(gcode_var.get() or "").strip()

        dialog = tk.Toplevel(self.root)
        dialog.title("保存参数配置")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        result = {"action": "session"}
        ttk.Label(dialog, text="参数辨识已完成。请选择配置保存方式：", font=UI_FONT_NORMAL).pack(anchor="w", padx=12, pady=(12, 8))
        if has_overwrite_target:
            detail = f"当前配置文件: {os.path.basename(active_path)}"
        elif gcode_path:
            detail = f"当前G代码: {os.path.basename(gcode_path)}；可另存为新配置并自动绑定。"
        else:
            detail = "当前尚未绑定配置文件，建议另存为新配置。"
        ttk.Label(dialog, text=detail, font=UI_FONT_SMALL).pack(anchor="w", padx=12, pady=(0, 10))

        btn_frame = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        btn_frame.pack(fill=tk.X)

        if has_overwrite_target:
            ttk.Button(
                btn_frame,
                text="覆盖当前配置",
                width=14,
                command=lambda: (result.update(action="overwrite"), dialog.destroy()),
            ).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(
            btn_frame,
            text="另存为",
            width=10,
            command=lambda: (result.update(action="save_as"), dialog.destroy()),
        ).pack(side=tk.LEFT)
        ttk.Button(
            btn_frame,
            text="仅本次使用",
            width=12,
            command=lambda: (result.update(action="session"), dialog.destroy()),
        ).pack(side=tk.RIGHT)

        center_dialog_on_parent(dialog, self.root)
        self.root.wait_window(dialog)

        action = result.get("action")
        if action == "overwrite":
            return bool(self.save_active_kc_case_profile(save_as=False))
        if action == "save_as":
            return bool(self.save_active_kc_case_profile(save_as=True))
        self.active_kc_profile_path = ""
        if self._get_profile_origin() == "runtime_identified_profile":
            self.kc_profile_status_var.set("案例配置: 当前运行时辨识结果（未保存）")
        else:
            self.kc_profile_status_var.set("当前内存配置（未保存）")
        self._debug_prediction_state_event(
            "profile_lock_disabled",
            reason="session_only",
            kc_map_source="runtime_fit" if self._get_profile_origin() == "runtime_identified_profile" else "current_rows",
        )
        self.set_status("参数辨识已完成，当前配置仅保留在本次会话内", 4000)
        return False

    def on_identification_mode_changed(self):
        self._refresh_lock_ke_check_text()
        self._refresh_lock_idle_check_text()
        self._persist_app_config()
        if hasattr(self, "_update_manual_kcke_button_text"):
            self._update_manual_kcke_button_text()

        fixed_ke = self._resolve_fixed_ke_for_identification()
        fixed_idle = self._resolve_fixed_global_idle_power()
        status_parts = []
        if fixed_ke is not None:
            status_parts.append(f"锁定全局K_e={fixed_ke:.6f}，仅更新K_c")
        elif getattr(self, "lock_ke_during_identification", None) is not None and self.lock_ke_during_identification.get():
            status_parts.append("已勾选锁定K_e，但当前全局K_e为空；下次仍会联合辨识K_c/K_e")
        else:
            status_parts.append("后续参数辨识将联合更新K_c/K_e")

        if fixed_idle is not None:
            status_parts.append(f"锁定全局P_idle={fixed_idle:.6f}")
        elif getattr(self, "lock_idle_during_identification", None) is not None and self.lock_idle_during_identification.get():
            status_parts.append("已勾选锁定P_idle，但当前全局P_idle为空；首次仍可写入")
        else:
            status_parts.append("后续参数辨识可更新全局P_idle")

        self.set_status("；".join(status_parts), 5000)

    def _fit_kc_ke_by_linear_system(self, mrr_values, ap_values, y_values, source_label, fixed_ke=None):
        mrr_arr = pd.to_numeric(pd.Series(mrr_values), errors='coerce').to_numpy(dtype=float)
        ap_arr = pd.to_numeric(pd.Series(ap_values), errors='coerce').to_numpy(dtype=float)
        y_arr = pd.to_numeric(pd.Series(y_values), errors='coerce').to_numpy(dtype=float)

        finite_mask = np.isfinite(mrr_arr) & np.isfinite(ap_arr) & np.isfinite(y_arr) & (mrr_arr > 1e-12)
        if fixed_ke is None:
            finite_mask = finite_mask & (ap_arr > 1e-12)
        mrr_arr = mrr_arr[finite_mask]
        ap_arr = ap_arr[finite_mask]
        y_arr = y_arr[finite_mask]

        if len(mrr_arr) < 2:
            target_label = "K_c" if fixed_ke is not None else "K_c / K_e"
            raise ValueError(f"{source_label}有效样本不足 2 个，无法辨识 {target_label}")

        if fixed_ke is not None:
            fixed_ke = max(float(fixed_ke), 0.0)
            design_matrix = mrr_arr.reshape(-1, 1)
            adjusted_y = y_arr - fixed_ke * ap_arr
            kc_value = self._solve_nonnegative_scalar(mrr_arr, adjusted_y)
            ke_value = fixed_ke
            residuals = y_arr - (kc_value * mrr_arr + ke_value * ap_arr)
            sample_count = len(y_arr)
            if sample_count > 1:
                residual_var = float(np.sum(residuals ** 2) / max(sample_count - 1, 1))
                cov_matrix = residual_var * np.linalg.pinv(design_matrix.T @ design_matrix)
                kc_sigma = math.sqrt(max(float(cov_matrix[0, 0]), 0.0))
            else:
                kc_sigma = 0.0

            return {
                "kc_value": float(kc_value),
                "ke_value": float(ke_value),
                "kc_sigma": float(max(kc_sigma, 0.0)),
                "kc_ucb": float(kc_value + float(self.kc_beta.get()) * max(kc_sigma, 0.0)),
                "interval_count": int(sample_count),
                "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
                "condition_number": 1.0,
                "ke_locked": True,
                "identification_mode": "kc_only",
            }

        design_matrix = np.column_stack([mrr_arr, ap_arr])
        scale_mrr = max(float(np.median(np.abs(mrr_arr))), 1.0)
        scale_ap = max(float(np.median(np.abs(ap_arr))), 1.0)
        scaled_design = np.column_stack([mrr_arr / scale_mrr, ap_arr / scale_ap])

        matrix_rank = int(np.linalg.matrix_rank(scaled_design))
        if matrix_rank < 2:
            specific_mrr = np.divide(
                mrr_arr,
                ap_arr,
                out=np.full_like(mrr_arr, np.nan, dtype=float),
                where=np.abs(ap_arr) > 1e-12,
            )
            excitation_span = float(np.nanmax(specific_mrr) - np.nanmin(specific_mrr)) if specific_mrr.size else 0.0
            excitation_scale = max(float(np.nanmedian(np.abs(specific_mrr))) if specific_mrr.size else 0.0, 1.0)
            excitation_ratio = excitation_span / excitation_scale
            if excitation_span <= 1e-6 or excitation_ratio < 0.02:
                raise ValueError(
                    f"{source_label}的 a_e·F/60 变化不足，当前数据主要只有 a_p 在变化，无法分离 K_c / K_e"
                )
            raise ValueError(f"{source_label}的 MRR 与 a_p 高度共线，无法稳定分离 K_c / K_e")

        condition_number = float(np.linalg.cond(scaled_design))
        if not np.isfinite(condition_number) or condition_number > 3e3:
            raise ValueError(f"{source_label}病态过强（条件数={condition_number:.1f}），无法稳定辨识 K_c / K_e")

        kc_value, ke_value = self._solve_nonnegative_kc_ke(mrr_arr, ap_arr, y_arr)
        coeffs = np.array([kc_value, ke_value], dtype=float)
        y_pred = design_matrix @ coeffs
        residuals = y_arr - y_pred

        sample_count = len(y_arr)
        if sample_count > 2:
            residual_var = float(np.sum(residuals ** 2) / max(sample_count - 2, 1))
            cov_matrix = residual_var * np.linalg.pinv(design_matrix.T @ design_matrix)
            kc_sigma = math.sqrt(max(float(cov_matrix[0, 0]), 0.0))
        else:
            kc_sigma = 0.0

        return {
            "kc_value": float(kc_value),
            "ke_value": float(ke_value),
            "kc_sigma": float(max(kc_sigma, 0.0)),
            "kc_ucb": float(kc_value + float(self.kc_beta.get()) * max(kc_sigma, 0.0)),
            "interval_count": int(sample_count),
            "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
            "condition_number": float(condition_number),
            "ke_locked": False,
            "identification_mode": "kc_ke",
        }

    def _get_sample_x_range_blocks(self, start_x, end_x, program_no=None):
        """基于样本 x 坐标快速定位区间内的连续样本块。"""
        if self.sample_data_x_positions is None:
            return []

        sample_x = np.asarray(self.sample_data_x_positions, dtype=float)
        if sample_x.size == 0:
            return []

        program_numbers = self.sample_data_program_numbers
        base_blocks = self.sample_data_base_blocks or [(0, len(sample_x) - 1)]
        result_blocks = []

        for block_start, block_end in base_blocks:
            block_start = int(block_start)
            block_end = int(block_end)
            if block_end < block_start:
                continue

            if program_no is not None and program_numbers is not None:
                try:
                    if program_numbers[block_start] != program_no:
                        continue
                except Exception:
                    continue

            block_x = sample_x[block_start:block_end + 1]
            if block_x.size == 0:
                continue

            if block_x.size > 1 and np.any(block_x[1:] < block_x[:-1]):
                block_mask = (block_x >= float(start_x)) & (block_x < float(end_x))
                for local_start, local_end in self.compute_contiguous_blocks(block_mask):
                    result_blocks.append((block_start + int(local_start), block_start + int(local_end)))
                continue

            left = int(np.searchsorted(block_x, float(start_x), side="left"))
            right = int(np.searchsorted(block_x, float(end_x), side="left"))
            if right > left:
                result_blocks.append((block_start + left, block_start + right - 1))

        return result_blocks

    def _update_measurement_gate_indicators(self, sigma_idle, delta_mrr, idle_count):
        if hasattr(self, "sigma_idle_var"):
            self.sigma_idle_var.set(self._format_metric_text(sigma_idle, "W"))
        if hasattr(self, "delta_mrr_var"):
            self.delta_mrr_var.set(self._format_metric_text(delta_mrr))
        if hasattr(self, "steady_gate_status_var"):
            try:
                idle_count_int = int(idle_count)
            except Exception:
                idle_count_int = 0
            if idle_count_int > 0:
                self.steady_gate_status_var.set(
                    f"稳态门控: 空载窗口 {idle_count_int} 点，方差阈值={(3.0 * float(sigma_idle)) ** 2:.6f}"
                )
            else:
                self.steady_gate_status_var.set("稳态门控: 未找到可用空载窗口，已退化为仅几何门控")

    def _estimate_idle_sigma_and_delta_mrr(self, sample_df, kc_reference=None):
        mrr_values = pd.to_numeric(sample_df["mrr"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        ap_values = pd.to_numeric(sample_df["ap"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        ae_values = pd.to_numeric(sample_df["ae"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        feed_values = pd.to_numeric(sample_df["feed_speed"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        actual_load = pd.to_numeric(sample_df["actual_load"], errors='coerce').to_numpy(dtype=float)
        idle_power = pd.to_numeric(sample_df["idle_power"], errors='coerce').to_numpy(dtype=float)

        idle_mask = (
            np.isfinite(actual_load)
            & np.isfinite(idle_power)
            & (
                (actual_load <= idle_power + 1e-9)
                | (mrr_values <= 1e-9)
                | (ap_values <= 1e-9)
                | (ae_values <= 1e-9)
                | (feed_values <= 1e-9)
            )
        )
        if int(np.sum(idle_mask)) < 20:
            finite_mrr = mrr_values[np.isfinite(mrr_values)]
            if finite_mrr.size:
                low_mrr_cutoff = float(np.percentile(finite_mrr, 10))
                idle_mask = (
                    np.isfinite(actual_load)
                    & np.isfinite(idle_power)
                    & (
                        (actual_load <= idle_power + 1e-9)
                        | (mrr_values <= max(low_mrr_cutoff, 1e-9))
                    )
                )

        residuals = actual_load[idle_mask] - idle_power[idle_mask]
        residuals = residuals[np.isfinite(residuals)]
        if residuals.size >= 5:
            low = float(np.percentile(residuals, 5))
            high = float(np.percentile(residuals, 95))
            trimmed = residuals[(residuals >= low) & (residuals <= high)]
            sigma_idle = self._robust_sigma(trimmed if trimmed.size >= 3 else residuals)
        elif residuals.size >= 2:
            sigma_idle = self._robust_sigma(residuals)
        else:
            sigma_idle = 0.0

        if not np.isfinite(sigma_idle) or sigma_idle < 0:
            sigma_idle = 0.0
        try:
            kc_ref = abs(float(kc_reference))
        except Exception:
            kc_ref = abs(self._resolve_measurement_gate_reference_kc())
        if kc_ref <= 1e-12:
            kc_ref = 1.0
        gamma = 3.0
        delta_mrr = gamma * sigma_idle / kc_ref if sigma_idle > 0 else 0.0
        return float(sigma_idle), float(delta_mrr), int(residuals.size), idle_mask

    def _append_manual_measurement_impedance(self, sample_df, sigma_idle, delta_mrr, idle_mask):
        sample_df = sample_df.copy()
        ap_values = pd.to_numeric(sample_df["ap"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        mrr_values = pd.to_numeric(sample_df["mrr"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        actual_load = pd.to_numeric(sample_df["actual_load"], errors='coerce').to_numpy(dtype=float)
        idle_power = pd.to_numeric(sample_df["idle_power"], errors='coerce').to_numpy(dtype=float)
        ke_value = self.get_ke_value()
        anchor_mask = np.asarray(
            sample_df["process_anchor_mask"] if "process_anchor_mask" in sample_df.columns else np.ones(len(sample_df), dtype=bool),
            dtype=bool,
        )

        kc_numerator = actual_load - idle_power - ke_value * ap_values
        process_idle_mask = (
            (mrr_values <= 1e-12)
            | (ap_values <= 1e-12)
        )
        actual_idle_mask = (
            np.isfinite(actual_load)
            & np.isfinite(idle_power)
            & (actual_load <= idle_power + 1e-9)
        )
        idle_point_mask = process_idle_mask | actual_idle_mask | np.asarray(idle_mask, dtype=bool)
        actual_series = pd.Series(actual_load, dtype=float)
        rolling_center = actual_series.rolling(window=7, center=True, min_periods=1).median().to_numpy(dtype=float)
        residual_from_center = np.abs(actual_load - rolling_center)
        prev_load = np.roll(actual_load, 1)
        next_load = np.roll(actual_load, -1)
        prev_load[0] = actual_load[0]
        next_load[-1] = actual_load[-1]
        max_jump = np.maximum(np.abs(actual_load - prev_load), np.abs(next_load - actual_load))
        cutting_ref = np.maximum(np.abs(rolling_center - idle_power), 0.0)
        spike_level_tol = np.maximum(6.0 * float(sigma_idle or 0.0), np.maximum(80.0, 0.35 * cutting_ref))
        spike_jump_tol = np.maximum(8.0 * float(sigma_idle or 0.0), np.maximum(120.0, 0.45 * cutting_ref))
        transition_spike_mask = (
            np.isfinite(actual_load)
            & np.isfinite(rolling_center)
            & (~idle_point_mask)
            & (
                (residual_from_center > spike_level_tol)
                | (max_jump > spike_jump_tol)
            )
        )
        base_valid = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            & np.isfinite(kc_numerator)
            & np.isfinite(mrr_values)
            & (mrr_values > 1e-12)
            & anchor_mask
            & ~idle_point_mask
            & ~transition_spike_mask
        )
        kc_valid_mask = base_valid & (mrr_values >= max(float(delta_mrr), 0.0))
        sample_kc_valid_mask = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            & np.isfinite(kc_numerator)
            & np.isfinite(mrr_values)
            & (mrr_values > 1e-12)
            & ~idle_point_mask
        )
        sample_kc_values = np.full(len(sample_df), np.nan, dtype=float)
        sample_kc_values[sample_kc_valid_mask] = kc_numerator[sample_kc_valid_mask] / mrr_values[sample_kc_valid_mask]
        sample_kc_values = self._clip_nonnegative_numeric_array(sample_kc_values)
        kc_point_values = np.full(len(sample_df), np.nan, dtype=float)
        kc_point_values[kc_valid_mask] = sample_kc_values[kc_valid_mask]

        sample_df["idle_window"] = np.asarray(idle_mask, dtype=bool)
        sample_df["is_idle_point"] = idle_point_mask
        sample_df["transition_spike"] = transition_spike_mask
        sample_df["kc_numerator"] = kc_numerator
        sample_df["sample_kc"] = sample_kc_values
        sample_df["sample_kc_valid"] = sample_kc_valid_mask
        sample_df["kc_point"] = kc_point_values
        sample_df["kc_valid"] = kc_valid_mask
        sample_df["kc_gated_out"] = base_valid & ~kc_valid_mask
        predicted_source = (
            sample_df["predicted_kc_source"].to_numpy(dtype=object)
            if "predicted_kc_source" in sample_df.columns else np.full(len(sample_df), "", dtype=object)
        )
        use_direct_sample_kc = np.asarray([str(value or "").strip() == "" for value in predicted_source], dtype=bool)
        if "predicted_kc" not in sample_df.columns:
            sample_df["predicted_kc"] = np.nan
        if "predicted_kc_source" not in sample_df.columns:
            sample_df["predicted_kc_source"] = ""
        if "predicted_load" not in sample_df.columns:
            sample_df["predicted_load"] = np.nan
        if np.any(use_direct_sample_kc):
            predicted_load = np.asarray(idle_power, dtype=float).copy()
            predicted_load[sample_kc_valid_mask] = (
                idle_power[sample_kc_valid_mask]
                + sample_kc_values[sample_kc_valid_mask] * mrr_values[sample_kc_valid_mask]
                + ke_value * ap_values[sample_kc_valid_mask]
            )
            predicted_load = np.maximum(predicted_load, 0.0)
            sample_df.loc[use_direct_sample_kc, "predicted_kc"] = sample_kc_values[use_direct_sample_kc]
            sample_df.loc[use_direct_sample_kc, "predicted_load"] = predicted_load[use_direct_sample_kc]
            sample_df.loc[use_direct_sample_kc & sample_kc_valid_mask, "predicted_kc_source"] = "measurement_point_kc"
        sample_df.loc[idle_point_mask, "predicted_load"] = np.maximum(idle_power[idle_point_mask], 0.0)
        sample_df["sigma_idle"] = float(sigma_idle)
        sample_df["delta_mrr"] = float(delta_mrr)
        return self._initialize_measurement_prediction_channels(sample_df)

    def _build_process_geometry_lookup(self):
        """按原始程序行号汇总工艺参数，供实验实测样本映射。"""
        lookup = {}
        for row_idx, row in enumerate(self.data or []):
            raw_key = self._get_process_row_sample_line(row, fallback=row_idx)

            bucket = lookup.setdefault(raw_key, {
                "ap": [],
                "ae": [],
                "line_no_aligned": [],
                "feed_plan": [],
                "speed_plan": [],
            })
            for key in ("ap", "ae", "line_no_aligned", "S"):
                value = row.get(key)
                try:
                    numeric = float(value)
                except Exception:
                    continue
                if not np.isfinite(numeric):
                    continue
                if key == "ap":
                    bucket["ap"].append(numeric)
                elif key == "ae":
                    bucket["ae"].append(numeric)
                elif key == "line_no_aligned":
                    bucket["line_no_aligned"].append(numeric)
                elif key == "S":
                    bucket["speed_plan"].append(numeric)

            feed_value = row.get("F_program")
            if feed_value is None:
                feed_value = row.get("feed_effective", row.get("F_plan"))
            try:
                feed_numeric = float(feed_value)
            except Exception:
                feed_numeric = float("nan")
            if np.isfinite(feed_numeric) and feed_numeric >= 0.0:
                bucket["feed_plan"].append(feed_numeric)

        records = []
        for raw_line, bucket in lookup.items():
            ap_values = bucket["ap"]
            ae_values = bucket["ae"]
            if not ap_values or not ae_values:
                continue
            records.append({
                "line_no_raw": int(raw_line),
                "line_no_aligned": int(round(float(np.median(bucket["line_no_aligned"])))) if bucket["line_no_aligned"] else int(raw_line),
                "ap": float(np.median(ap_values)),
                "ae": float(np.median(ae_values)),
                "feed_plan": float(np.median(bucket["feed_plan"])) if bucket["feed_plan"] else np.nan,
                "speed_plan": float(np.median(bucket["speed_plan"])) if bucket["speed_plan"] else np.nan,
            })

        if not records:
            raise ValueError("当前工艺信息文件中未找到可映射的 ap/ae 数据")

        return pd.DataFrame.from_records(records)

    def _build_process_point_lookup(self):
        """按原始程序行号保留工艺点序列，供实测点在同一行内均匀对齐。"""
        # profile 应用会统一失效对齐缓存；缓存键不得反向解析 profile，
        # 否则 profile 规范化在投影端点时会递归进入本方法。
        cache_key = (
            id(self.data),
            len(self.data or []),
            int(getattr(self, "_process_model_state_version", 0) or 0),
            self._get_prediction_source(),
            str(getattr(self, "step_feed_model_signature", "") or ""),
            float(self.get_kc_value()),
            float(self.get_ke_value()),
        )
        if getattr(self, "_process_point_lookup_cache_key", None) == cache_key:
            cached_lookup = getattr(self, "_process_point_lookup_cache", None)
            if cached_lookup:
                return cached_lookup

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return float("nan")
            return numeric if np.isfinite(numeric) else float("nan")

        lookup = {}
        for row_idx, row in enumerate(self.data or []):
            raw_key = self._get_process_row_sample_line(row, fallback=row_idx)

            ap_val = _safe_float(row.get("ap"))
            ae_val = _safe_float(row.get("ae"))
            aligned_val = _safe_float(row.get("line_no_aligned", raw_key))
            speed_val = _safe_float(row.get("S"))
            feed_value = row.get("F_program")
            if feed_value is None:
                feed_value = row.get("feed_effective", row.get("F_plan"))
            feed_plan = _safe_float(feed_value)
            kc_val = _safe_float(row.get("K_c", row.get("K", self.get_kc_value())))
            if not np.isfinite(feed_plan) or feed_plan < 0.0:
                feed_plan = 0.0
            process_mrr = 0.0
            if np.isfinite(ap_val) and np.isfinite(ae_val) and ap_val > 1e-12 and ae_val > 1e-12 and feed_plan > 1e-12:
                process_mrr = float(ap_val * ae_val * feed_plan / 60.0)

            bucket = lookup.setdefault(raw_key, {
                "line_no_aligned": [],
                "ap": [],
                "ae": [],
                "feed_plan": [],
                "speed_plan": [],
                "process_mrr": [],
                "process_kc": [],
                "process_row_index": [],
                "process_point_index": [],
            })
            bucket["line_no_aligned"].append(int(round(float(aligned_val))) if np.isfinite(aligned_val) else int(raw_key))
            bucket["ap"].append(float(ap_val) if np.isfinite(ap_val) else np.nan)
            bucket["ae"].append(float(ae_val) if np.isfinite(ae_val) else np.nan)
            bucket["feed_plan"].append(float(feed_plan) if np.isfinite(feed_plan) else np.nan)
            bucket["speed_plan"].append(float(speed_val) if np.isfinite(speed_val) else np.nan)
            bucket["process_mrr"].append(float(process_mrr) if np.isfinite(process_mrr) else np.nan)
            bucket["process_kc"].append(float(kc_val) if np.isfinite(kc_val) else float(self.get_kc_value()))
            bucket["process_row_index"].append(int(row_idx))
            bucket["process_point_index"].append(int(row.get("process_point_index", len(bucket["line_no_aligned"]) - 1) or 0))

        if not lookup:
            raise ValueError("当前工艺信息文件中未找到可映射的工艺点")

        for raw_line, bucket in lookup.items():
            for key in (
                "line_no_aligned",
                "ap",
                "ae",
                "feed_plan",
                "speed_plan",
                "process_mrr",
                "process_kc",
                "process_row_index",
                "process_point_index",
            ):
                bucket[key] = np.asarray(bucket[key])
            bucket["point_count"] = int(len(bucket["ap"]))
            point_count = max(int(bucket["point_count"]), 1)
            bucket["process_anchor_x"] = (
                float(raw_line)
                + bucket["process_point_index"].astype(float) / float(point_count)
            )
        self._process_point_lookup_cache = lookup
        self._process_point_lookup_cache_key = cache_key
        return lookup

    def _build_aligned_process_geometry_frame(self, raw_line_numbers):
        """在同一原始程序行内，将实测点映射到工艺点序列，并标记稀疏反解锚点。"""
        raw_lines = np.asarray(raw_line_numbers, dtype=int)
        if raw_lines.size == 0:
            return pd.DataFrame({
                "line_no_aligned": np.asarray([], dtype=int),
                "ap": np.asarray([], dtype=float),
                "ae": np.asarray([], dtype=float),
                "feed_plan": np.asarray([], dtype=float),
                "speed_plan": np.asarray([], dtype=float),
                "process_mrr": np.asarray([], dtype=float),
                "process_kc": np.asarray([], dtype=float),
                "process_point_index": np.asarray([], dtype=int),
                "process_point_count": np.asarray([], dtype=int),
                "process_row_index": np.asarray([], dtype=int),
                "process_point_anchor_x": np.asarray([], dtype=float),
                "sample_anchor_x": np.asarray([], dtype=float),
                "process_anchor_mask": np.asarray([], dtype=bool),
                "process_anchor_index": np.asarray([], dtype=int),
            })

        point_lookup = self._build_process_point_lookup()
        sample_count = len(raw_lines)
        aligned_lines = np.full(sample_count, np.nan, dtype=float)
        ap_values = np.full(sample_count, np.nan, dtype=float)
        ae_values = np.full(sample_count, np.nan, dtype=float)
        feed_plan_values = np.full(sample_count, np.nan, dtype=float)
        speed_plan_values = np.full(sample_count, np.nan, dtype=float)
        process_mrr_values = np.full(sample_count, np.nan, dtype=float)
        process_kc_values = np.full(sample_count, np.nan, dtype=float)
        process_point_index = np.full(sample_count, -1, dtype=int)
        process_point_count = np.zeros(sample_count, dtype=int)
        process_row_index = np.full(sample_count, -1, dtype=int)
        process_point_anchor_x = np.full(sample_count, np.nan, dtype=float)
        sample_anchor_x = np.full(sample_count, np.nan, dtype=float)
        process_anchor_mask = np.zeros(sample_count, dtype=bool)
        process_anchor_index = np.full(sample_count, -1, dtype=int)

        start = 0
        while start < sample_count:
            raw_line = int(raw_lines[start])
            end = start + 1
            while end < sample_count and int(raw_lines[end]) == raw_line:
                end += 1

            bucket = point_lookup.get(raw_line)
            if bucket:
                point_count = int(bucket.get("point_count", 0) or 0)
                if point_count > 0:
                    local_count = end - start
                    mapped_index = np.floor((np.arange(local_count, dtype=float) + 0.5) * point_count / float(local_count)).astype(int)
                    mapped_index = np.clip(mapped_index, 0, point_count - 1)
                    target_slice = slice(start, end)
                    aligned_lines[target_slice] = bucket["line_no_aligned"][mapped_index]
                    ap_values[target_slice] = bucket["ap"][mapped_index]
                    ae_values[target_slice] = bucket["ae"][mapped_index]
                    feed_plan_values[target_slice] = bucket["feed_plan"][mapped_index]
                    speed_plan_values[target_slice] = bucket["speed_plan"][mapped_index]
                    process_mrr_values[target_slice] = bucket["process_mrr"][mapped_index]
                    process_kc_values[target_slice] = bucket["process_kc"][mapped_index]
                    process_point_index[target_slice] = bucket["process_point_index"][mapped_index]
                    process_point_count[target_slice] = point_count
                    process_row_index[target_slice] = bucket["process_row_index"][mapped_index]
                    process_point_anchor_x[target_slice] = bucket["process_anchor_x"][mapped_index]
                    sample_anchor_x[target_slice] = float(raw_line) + np.arange(local_count, dtype=float) / float(local_count)
                    if local_count >= point_count:
                        anchor_offsets = np.floor(np.arange(point_count, dtype=float) * local_count / float(point_count)).astype(int)
                        anchor_offsets = np.clip(anchor_offsets, 0, local_count - 1)
                        anchor_indices = np.arange(point_count, dtype=int)
                    else:
                        anchor_offsets = np.arange(local_count, dtype=int)
                        anchor_indices = np.floor(
                            (np.arange(local_count, dtype=float) + 0.5) * point_count / float(local_count)
                        ).astype(int)
                        anchor_indices = np.clip(anchor_indices, 0, point_count - 1)
                    anchor_pairs = {}
                    for local_offset, anchor_idx in zip(anchor_offsets, anchor_indices):
                        anchor_pairs.setdefault(int(local_offset), int(anchor_idx))
                    for local_offset in sorted(anchor_pairs.keys()):
                        absolute_idx = int(start + local_offset)
                        process_anchor_mask[absolute_idx] = True
                        process_anchor_index[absolute_idx] = int(anchor_pairs[local_offset])
            else:
                target_slice = slice(start, end)
                aligned_lines[target_slice] = float(raw_line)
                ap_values[target_slice] = 0.0
                ae_values[target_slice] = 0.0
                feed_plan_values[target_slice] = 0.0
                speed_plan_values[target_slice] = np.nan
                process_mrr_values[target_slice] = 0.0
                process_kc_values[target_slice] = float(self.get_kc_value())
                process_point_index[target_slice] = -1
                process_point_count[target_slice] = 0
                process_anchor_mask[target_slice] = False
                process_anchor_index[target_slice] = -1
            start = end

        return pd.DataFrame({
            "line_no_aligned": aligned_lines,
            "ap": ap_values,
            "ae": ae_values,
            "feed_plan": feed_plan_values,
            "speed_plan": speed_plan_values,
            "process_mrr": process_mrr_values,
            "process_kc": process_kc_values,
            "process_point_index": process_point_index,
            "process_point_count": process_point_count,
            "process_row_index": process_row_index,
            "process_point_anchor_x": process_point_anchor_x,
            "sample_anchor_x": sample_anchor_x,
            "process_anchor_mask": process_anchor_mask,
            "process_anchor_index": process_anchor_index,
        })

    def _build_manual_measurement_idle_speed_reference(self, spindle_speed_values):
        """为实验实测构造用于空载重建的转速参考。

        保留真实的 0 -> 升速 -> 稳态 -> 降速 -> 0 轮廓，
        仅对正转速区间内部的短 0/NaN 空洞做线性插值，避免后验曲线无故频繁砸到 0。
        """
        speed_arr = pd.to_numeric(pd.Series(spindle_speed_values), errors='coerce').to_numpy(dtype=float)
        if speed_arr.size == 0:
            return speed_arr

        speed_arr[~np.isfinite(speed_arr)] = np.nan
        positive_mask = speed_arr > 1e-9
        ref_arr = speed_arr.copy()
        ref_arr[~positive_mask] = np.nan

        speed_series = pd.Series(ref_arr)
        try:
            ref_arr = speed_series.interpolate(method="linear", limit_area="inside").to_numpy(dtype=float)
        except TypeError:
            first_valid = speed_series.first_valid_index()
            last_valid = speed_series.last_valid_index()
            ref_arr = speed_series.interpolate(method="linear").to_numpy(dtype=float)
            if first_valid is not None:
                ref_arr[:int(first_valid)] = np.nan
            if last_valid is not None:
                ref_arr[int(last_valid) + 1:] = np.nan

        ref_arr = np.nan_to_num(ref_arr, nan=0.0, posinf=0.0, neginf=0.0)
        ref_arr[ref_arr < 0.0] = 0.0
        return ref_arr

    def _build_manual_measurement_sample_frame(self, allow_saved_sample_profile=True):
        """将实验实测与工艺信息按程序行号对齐，生成样本级预测负载数据表。"""
        measurement = getattr(self, "manual_measurement_data", None)
        if not measurement:
            raise ValueError("未加载实验实测文件")

        sample_df = pd.DataFrame({
            "sample_index": np.arange(len(measurement["program_line"]), dtype=int),
            "line_no_raw": np.asarray(measurement["program_line"], dtype=int),
            "actual_load": np.abs(np.asarray(measurement["actual_load"], dtype=float)),
            "spindle_speed_actual": np.asarray(measurement.get("actual_spindle_speed", np.zeros(len(measurement["program_line"]), dtype=float)), dtype=float),
            "feed_speed_actual": np.asarray(measurement.get("actual_feed_speed", np.zeros(len(measurement["program_line"]), dtype=float)), dtype=float),
        })
        process_df = self._build_aligned_process_geometry_frame(sample_df["line_no_raw"].to_numpy(dtype=int))
        sample_df = pd.concat([sample_df.reset_index(drop=True), process_df.reset_index(drop=True)], axis=1)
        sample_df["line_no_aligned"] = sample_df["line_no_aligned"].fillna(sample_df["line_no_raw"])
        actual_spindle = pd.to_numeric(sample_df["spindle_speed_actual"], errors='coerce').to_numpy(dtype=float)
        speed_plan = pd.to_numeric(sample_df["speed_plan"], errors='coerce').to_numpy(dtype=float)
        sample_df["spindle_speed"] = np.where(
            np.isfinite(actual_spindle) & (actual_spindle > 1e-9),
            actual_spindle,
            speed_plan,
        )
        idle_speed_ref = self._build_manual_measurement_idle_speed_reference(sample_df["spindle_speed"])
        sample_df["idle_speed_ref"] = idle_speed_ref
        idle_power = np.zeros(len(sample_df), dtype=float)
        positive_idle_mask = np.isfinite(idle_speed_ref) & (idle_speed_ref > 1e-9)
        if np.any(positive_idle_mask):
            idle_predictor = self._create_idle_power_predictor() if hasattr(self, "_create_idle_power_predictor") else self.predict_idle_power
            speed_values = np.asarray(idle_speed_ref[positive_idle_mask], dtype=float)
            rounded_speeds = np.round(speed_values, 6)
            unique_speeds, inverse_indices = np.unique(rounded_speeds, return_inverse=True)
            unique_idle_power = np.asarray([idle_predictor(speed) for speed in unique_speeds], dtype=float)
            idle_power[positive_idle_mask] = unique_idle_power[inverse_indices]
        sample_df["idle_power"] = idle_power
        geometry_available = (
            np.isfinite(pd.to_numeric(sample_df["ap"], errors='coerce').to_numpy(dtype=float))
            & np.isfinite(pd.to_numeric(sample_df["ae"], errors='coerce').to_numpy(dtype=float))
        )
        geometry_positive = (
            pd.to_numeric(sample_df["ap"], errors='coerce').to_numpy(dtype=float) > 1e-12
        ) & (
            pd.to_numeric(sample_df["ae"], errors='coerce').to_numpy(dtype=float) > 1e-12
        )
        feed_plan = pd.to_numeric(sample_df["feed_plan"], errors='coerce').to_numpy(dtype=float)
        program_feed = np.where(np.isfinite(feed_plan) & (feed_plan >= 0.0), feed_plan, 0.0)
        sample_df["feed_speed"] = program_feed
        sample_df["mrr"] = (
            pd.to_numeric(sample_df["ap"], errors='coerce')
            * pd.to_numeric(sample_df["ae"], errors='coerce')
            * pd.Series(program_feed, index=sample_df.index)
            / 60.0
        )

        ap_for_prediction = pd.to_numeric(sample_df["ap"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        mrr_for_prediction = pd.to_numeric(sample_df["mrr"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        ke_value = self._get_effective_ke_value_from_profile(default=self.get_ke_value())
        sample_df["predicted_kc"] = np.nan
        sample_df["predicted_load"] = np.nan
        sample_df["predicted_kc_source"] = ""
        if allow_saved_sample_profile and self._should_apply_saved_sample_profile_for_measurement_prediction():
            saved_sample_profile = self._resolve_saved_sample_kc_profile(measurement=measurement)
            if isinstance(saved_sample_profile, dict):
                saved_kc_values = self._clip_nonnegative_numeric_array(saved_sample_profile.get("kc_values", []))
                saved_valid_mask = np.asarray(saved_sample_profile.get("valid_mask", []), dtype=bool)
                if saved_kc_values.size == len(sample_df) and saved_valid_mask.size == len(sample_df):
                    predicted_load = np.asarray(idle_power, dtype=float).copy()
                    predicted_load[saved_valid_mask] = (
                        idle_power[saved_valid_mask]
                        + saved_kc_values[saved_valid_mask] * mrr_for_prediction[saved_valid_mask]
                        + ke_value * ap_for_prediction[saved_valid_mask]
                    )
                    predicted_load = np.maximum(predicted_load, 0.0)
                    sample_df["predicted_kc"] = saved_kc_values
                    sample_df["predicted_load"] = predicted_load
                    sample_df.loc[saved_valid_mask, "predicted_kc_source"] = str(
                        saved_sample_profile.get("source") or "sample_kc_profile"
                    )
        sample_df["cutting_load"] = sample_df["actual_load"] - sample_df["idle_power"]
        sample_df["prediction_valid"] = (
            np.isfinite(sample_df["actual_load"].to_numpy(dtype=float))
            & geometry_available
            & np.isfinite(sample_df["feed_speed"].to_numpy(dtype=float))
            & np.isfinite(sample_df["mrr"].to_numpy(dtype=float))
            & geometry_positive
            & (sample_df["mrr"].to_numpy(dtype=float) > 1e-12)
        )
        return sample_df

    def _get_measurement_display_mode(self, prediction_source=None):
        sample_mode = str(getattr(self, "sample_data_mode", "") or "").strip()
        if sample_mode != "experiment_measurement" or not getattr(self, "manual_measurement_data", None):
            return "forward"
        normalized_source = self._normalize_profile_origin(
            prediction_source if prediction_source is not None else self._get_prediction_source()
        )
        if normalized_source == "imported_profile":
            return "forward"
        if normalized_source == "runtime_identified_profile":
            return "posterior"
        return "posterior"

    def _initialize_measurement_prediction_channels(self, sample_df):
        if sample_df is None or sample_df.empty:
            return sample_df

        sample_df = sample_df.copy()
        row_count = len(sample_df)
        display_kc = pd.to_numeric(sample_df.get("predicted_kc"), errors="coerce").to_numpy(dtype=float)
        display_load = pd.to_numeric(sample_df.get("predicted_load"), errors="coerce").to_numpy(dtype=float)
        display_source = (
            sample_df["predicted_kc_source"].astype(str).to_numpy(dtype=object)
            if "predicted_kc_source" in sample_df.columns else np.full(row_count, "", dtype=object)
        )
        sample_df["display_predicted_kc"] = display_kc
        sample_df["display_predicted_load"] = display_load
        sample_df["display_prediction_source"] = display_source
        if "interval_summary_kc" not in sample_df.columns:
            sample_df["interval_summary_kc"] = display_kc.copy()
        if "interval_summary_load" not in sample_df.columns:
            sample_df["interval_summary_load"] = display_load.copy()
        if "interval_summary_source" not in sample_df.columns:
            sample_df["interval_summary_source"] = display_source.copy()
        return self._sync_display_prediction_aliases(sample_df)

    def _sync_display_prediction_aliases(self, sample_df):
        if sample_df is None or sample_df.empty:
            return sample_df

        sample_df = sample_df.copy()
        row_count = len(sample_df)
        display_kc = pd.to_numeric(
            sample_df.get("display_predicted_kc", sample_df.get("predicted_kc")),
            errors="coerce",
        ).to_numpy(dtype=float)
        display_load = pd.to_numeric(
            sample_df.get("display_predicted_load", sample_df.get("predicted_load")),
            errors="coerce",
        ).to_numpy(dtype=float)
        display_source = (
            sample_df["display_prediction_source"].astype(str).to_numpy(dtype=object)
            if "display_prediction_source" in sample_df.columns
            else (
                sample_df["predicted_kc_source"].astype(str).to_numpy(dtype=object)
                if "predicted_kc_source" in sample_df.columns else np.full(row_count, "", dtype=object)
            )
        )
        sample_df["display_predicted_kc"] = display_kc
        sample_df["display_predicted_load"] = display_load
        sample_df["display_prediction_source"] = display_source
        sample_df["predicted_kc"] = display_kc
        sample_df["predicted_load"] = display_load
        sample_df["predicted_kc_source"] = display_source
        return sample_df

    def _resolve_measurement_parameter_source(self, prediction_source=None):
        normalized_source = self._normalize_profile_origin(
            prediction_source if prediction_source is not None else self._get_prediction_source()
        )
        return normalized_source if normalized_source != "no_profile" else "current_rows"

    def _resolve_measurement_interval_source(self, interval_records=None, fallback="none"):
        if isinstance(interval_records, list) and interval_records:
            current_source = str(getattr(self, "_current_interval_source", "") or "").strip()
            if current_source:
                return current_source
            return str(fallback or "measurement_interval_summary")
        return "none"

    def _should_reverse_solve_measurement_prediction(
        self,
        allow_measurement_resolve=None,
        profile_origin=None,
        profile=None,
        allow_autoload_imported=None,
        display_mode=None,
    ):
        resolved_origin = profile_origin
        resolved_profile = profile
        resolved_display_mode = str(display_mode or "").strip()
        if resolved_origin is None:
            resolved_origin, resolved_profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload()
                if allow_autoload_imported is None else bool(allow_autoload_imported),
            )
        if resolved_origin == "imported_profile" and isinstance(resolved_profile, dict):
            reverse_solve = False
        elif resolved_display_mode == "posterior":
            reverse_solve = True
        elif allow_measurement_resolve is not None:
            reverse_solve = bool(allow_measurement_resolve)
        else:
            reverse_solve = not (
                resolved_origin in {"imported_profile", "runtime_identified_profile"}
                and isinstance(resolved_profile, dict)
            )
        self._debug_prediction_state_event(
            "resolve_reverse_solve",
            reverse_solve=bool(reverse_solve),
            display_mode=resolved_display_mode or self._get_measurement_display_mode(prediction_source=resolved_origin),
            kc_map_source=self._get_prediction_source() if self._get_prediction_source() != "no_profile" else "current_rows",
        )
        return bool(reverse_solve)

    def _build_interval_template_sample_kc_array(self, sample_df, profile=None):
        sample_size = int(len(sample_df)) if sample_df is not None else 0
        if sample_size <= 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=object)

        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return np.full(sample_size, np.nan, dtype=float), np.full(sample_size, "", dtype=object)

        try:
            raw_lines = (
                pd.to_numeric(sample_df.get("line_no_raw"), errors="coerce").fillna(-1).to_numpy(dtype=int)
                if "line_no_raw" in sample_df.columns
                else np.full(sample_size, -1, dtype=int)
            )
        except Exception:
            raw_lines = np.full(sample_size, -1, dtype=int)

        global_kc = self._resolve_profile_global_kc(source_profile, default=self.get_kc_value())
        template_kc = np.full(sample_size, np.nan, dtype=float)
        template_sources = np.full(sample_size, "", dtype=object)
        segment_records = self._extract_profile_segment_records(source_profile)
        for segment in segment_records:
            if not isinstance(segment, dict):
                continue
            is_idle_segment = (
                int(self._resolve_smif_state_code(segment)) == 1
                or bool(segment.get("is_idle_interval"))
                or str(segment.get("segment_type") or "").strip().lower() == "idle"
            )
            if not is_idle_segment:
                continue
            segment_mask = self._build_interval_sample_mask(
                segment,
                sample_size,
                line_numbers=raw_lines,
            )
            if not np.any(segment_mask):
                continue
            template_kc[segment_mask] = np.nan
            template_sources[segment_mask] = "profile_idle"

        pit_records = [
            dict(record)
            for record in (source_profile.get("pit_records", []) or [])
            if isinstance(record, dict)
        ]
        interval_records = self._extract_profile_interval_records(source_profile)
        interval_source_label = "profile_interval_mode" if pit_records else "interval_template"
        if not interval_records:
            return template_kc, template_sources

        for interval in interval_records:
            if not isinstance(interval, dict):
                continue
            interval_mask = self._build_interval_sample_mask(
                interval,
                sample_size,
                line_numbers=raw_lines,
            )
            if not np.any(interval_mask):
                continue
            is_idle_interval = bool(interval.get("is_idle_interval")) or str(interval.get("kc_source", "")).strip().lower() == "idle"
            if is_idle_interval:
                template_kc[interval_mask] = np.nan
                template_sources[interval_mask] = "profile_idle" if pit_records else "interval_template_idle"
                continue
            try:
                interval_kc = float(interval.get("K_c_hat"))
            except Exception:
                interval_kc = float("nan")
            if not np.isfinite(interval_kc):
                interval_kc = float(global_kc)
            if not np.isfinite(interval_kc):
                continue
            interval_kc = max(float(interval_kc), 0.0)
            template_kc[interval_mask] = interval_kc
            template_sources[interval_mask] = interval_source_label
        return template_kc, template_sources

    def _build_authoritative_sample_kc_array(self, sample_df, profile=None, prefer_interval_templates=False):
        if sample_df is None or sample_df.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=object)
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            return np.full(len(sample_df), np.nan, dtype=float), np.full(len(sample_df), "", dtype=object)

        line_kc_map = self._normalize_profile_line_kc_map(source_profile)
        point_kc_map = self._normalize_profile_point_kc_map(source_profile)
        global_kc = self._resolve_profile_global_kc(source_profile, default=self.get_kc_value())

        aligned_lines = pd.to_numeric(sample_df.get("line_no_aligned"), errors="coerce").to_numpy(dtype=float)
        point_indices = (
            pd.to_numeric(sample_df.get("process_point_index"), errors="coerce").fillna(-1).to_numpy(dtype=int)
            if "process_point_index" in sample_df.columns
            else np.full(len(sample_df), -1, dtype=int)
        )

        kc_values = np.full(len(sample_df), float(global_kc), dtype=float)
        kc_sources = np.full(len(sample_df), "global_kc", dtype=object)
        for idx in range(len(sample_df)):
            if idx >= aligned_lines.size or not np.isfinite(aligned_lines[idx]):
                continue
            line_no = int(round(float(aligned_lines[idx])))
            point_idx = int(point_indices[idx]) if idx < point_indices.size else -1
            if line_no in line_kc_map:
                kc_values[idx] = float(line_kc_map[line_no])
                kc_sources[idx] = "line_kc_map"
        for idx in range(len(sample_df)):
            if idx >= aligned_lines.size or not np.isfinite(aligned_lines[idx]):
                continue
            line_no = int(round(float(aligned_lines[idx])))
            point_idx = int(point_indices[idx]) if idx < point_indices.size else -1
            if point_idx >= 0 and (line_no, point_idx) in point_kc_map:
                kc_values[idx] = float(point_kc_map[(line_no, point_idx)])
                kc_sources[idx] = "point_kc_map"

        if self._should_apply_saved_sample_profile_for_measurement_prediction():
            saved_sample_profile = self._resolve_saved_sample_kc_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
            )
            if isinstance(saved_sample_profile, dict):
                saved_kc_values = self._clip_nonnegative_numeric_array(saved_sample_profile.get("kc_values", []))
                saved_valid_mask = np.asarray(saved_sample_profile.get("valid_mask", []), dtype=bool)
                if saved_kc_values.size == len(sample_df) and saved_valid_mask.size == len(sample_df):
                    kc_values[saved_valid_mask] = saved_kc_values[saved_valid_mask]
                    kc_sources[saved_valid_mask] = "sample_kc_profile"

        if bool(prefer_interval_templates):
            interval_kc, interval_sources = self._build_interval_template_sample_kc_array(
                sample_df,
                profile=source_profile,
            )
            idle_mask = interval_sources == "profile_idle"
            if np.any(idle_mask):
                kc_values[idle_mask] = np.nan
                kc_sources[idle_mask] = "profile_idle"
            interval_override_mask = np.isfinite(interval_kc) & (interval_sources == "profile_interval_mode")
            if np.any(interval_override_mask):
                kc_values[interval_override_mask] = interval_kc[interval_override_mask]
                kc_sources[interval_override_mask] = interval_sources[interval_override_mask]
            else:
                legacy_mask = np.isfinite(interval_kc) & (interval_sources != "profile_idle")
                if np.any(legacy_mask):
                    kc_values[legacy_mask] = interval_kc[legacy_mask]
                    kc_sources[legacy_mask] = interval_sources[legacy_mask]

        return kc_values, kc_sources

    def _resolve_forward_prediction_mrr_values(self, sample_df, preferred_column=None, profile=None, source_origin=""):
        if sample_df is None or sample_df.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=object), "computed_program_mrr", 0

        row_count = int(len(sample_df))
        ap_values = pd.to_numeric(sample_df["ap"], errors="coerce").to_numpy(dtype=float)
        ae_values = pd.to_numeric(sample_df["ae"], errors="coerce").to_numpy(dtype=float)
        feed_values = pd.to_numeric(sample_df["feed_plan"], errors="coerce").to_numpy(dtype=float)
        resolved_mrr = ap_values * ae_values * feed_values / 60.0
        valid_mask = (
            np.isfinite(ap_values)
            & np.isfinite(ae_values)
            & np.isfinite(feed_values)
            & np.isfinite(resolved_mrr)
            & (ap_values >= 0.0)
            & (ae_values >= 0.0)
            & (feed_values >= 0.0)
        )
        resolved_mrr = np.where(valid_mask, np.maximum(resolved_mrr, 0.0), 0.0)
        resolved_source = np.full(row_count, "computed_program_mrr", dtype=object)
        return resolved_mrr, resolved_source, "computed_program_mrr", int(np.sum(valid_mask))

    def _compute_forward_prediction_for_sample_df(
        self,
        sample_df,
        profile=None,
        *,
        forward_label="profile_forward",
        idle_label="profile_idle",
        prefer_interval_templates=False,
        preferred_mrr_column=None,
        source_origin="",
    ):
        if sample_df is None or sample_df.empty:
            return sample_df

        sample_df = sample_df.copy()
        row_count = len(sample_df)
        idle_power = pd.to_numeric(sample_df.get("idle_power"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ap_values = pd.to_numeric(sample_df.get("ap"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mrr_values, mrr_source_values, _mrr_source_label, _preferred_mrr_count = self._resolve_forward_prediction_mrr_values(
            sample_df,
            preferred_column=preferred_mrr_column,
            profile=profile,
            source_origin=source_origin,
        )
        prediction_valid = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            if "prediction_valid" in sample_df.columns else np.ones(row_count, dtype=bool)
        )
        authoritative_kc, authoritative_sources = self._build_authoritative_sample_kc_array(
            sample_df,
            profile=profile,
            prefer_interval_templates=bool(prefer_interval_templates),
        )
        finite_kc_mask = np.isfinite(authoritative_kc)
        authoritative_kc[finite_kc_mask] = np.maximum(authoritative_kc[finite_kc_mask], 0.0)
        cutting_mask = (
            prediction_valid
            & finite_kc_mask
            & np.isfinite(mrr_values)
            & np.isfinite(ap_values)
            & (mrr_values > 1e-12)
        )
        idle_point_mask = ~cutting_mask
        ke_value = self._resolve_profile_ke_value(profile, default=self.get_ke_value())

        predicted_kc = np.full(row_count, np.nan, dtype=float)
        predicted_load = np.asarray(idle_power, dtype=float).copy()
        predicted_source = np.asarray(authoritative_sources, dtype=object).copy()
        predicted_source[predicted_source == ""] = str(forward_label)
        predicted_source[idle_point_mask] = str(idle_label)
        if np.any(cutting_mask):
            predicted_kc[cutting_mask] = authoritative_kc[cutting_mask]
            predicted_load[cutting_mask] = (
                idle_power[cutting_mask]
                + authoritative_kc[cutting_mask] * mrr_values[cutting_mask]
                + ke_value * ap_values[cutting_mask]
            )
        predicted_load = np.maximum(predicted_load, 0.0)

        sample_df["authoritative_kc"] = authoritative_kc
        sample_df["authoritative_kc_source"] = authoritative_sources
        sample_df["forward_prediction_mrr"] = mrr_values
        sample_df["forward_prediction_mrr_source"] = mrr_source_values
        sample_df["predicted_kc"] = predicted_kc
        sample_df["predicted_load"] = predicted_load
        sample_df["predicted_kc_source"] = predicted_source
        sample_df["kc_point"] = np.full(row_count, np.nan, dtype=float)
        sample_df["sample_kc"] = np.full(row_count, np.nan, dtype=float)
        sample_df["sample_kc_valid"] = np.zeros(row_count, dtype=bool)
        sample_df["sample_kc_source"] = np.full(row_count, "", dtype=object)
        sample_df["kc_valid"] = np.zeros(row_count, dtype=bool)
        sample_df["kc_gated_out"] = np.zeros(row_count, dtype=bool)
        sample_df["idle_window"] = np.zeros(row_count, dtype=bool)
        sample_df["is_idle_point"] = idle_point_mask
        sample_df["transition_spike"] = np.zeros(row_count, dtype=bool)
        sample_df["kc_numerator"] = np.full(row_count, np.nan, dtype=float)
        sample_df["sigma_idle"] = 0.0
        sample_df["delta_mrr"] = 0.0
        return self._initialize_measurement_prediction_channels(sample_df)

    def _apply_profile_forward_prediction_to_sample_df(self, sample_df, profile=None, source_label="imported_profile"):
        if sample_df is None or sample_df.empty:
            return sample_df

        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            source_origin, source_profile = self._resolve_forward_prediction_profile(
                measurement=getattr(self, "manual_measurement_data", None),
                process_path=self._get_primary_input_file_or_empty(),
                allow_autoload_imported=self._should_allow_imported_profile_autoload(),
            )
        else:
            source_origin = str(source_label or "profile")

        forward_label = "runtime_forward" if source_origin == "runtime_identified_profile" else "profile_forward"
        idle_label = "runtime_idle" if source_origin == "runtime_identified_profile" else "profile_idle"
        prefer_interval_templates = False
        sample_df = self._compute_forward_prediction_for_sample_df(
            sample_df,
            profile=source_profile,
            forward_label=forward_label,
            idle_label=idle_label,
            prefer_interval_templates=prefer_interval_templates,
            preferred_mrr_column=None,
            source_origin=source_origin,
        )
        forward_mrr_source = (
            sample_df["forward_prediction_mrr_source"].astype(str).to_numpy(dtype=object)
            if "forward_prediction_mrr_source" in sample_df.columns else np.full(len(sample_df), "computed_program_mrr", dtype=object)
        )
        computed_program_mrr_count = int(np.sum(forward_mrr_source == "computed_program_mrr"))
        self._debug_prediction_state_event(
            "apply_profile_forward_prediction",
            reverse_solve=False,
            prefer_interval_templates=prefer_interval_templates,
            mrr_source="computed_program_mrr",
            computed_program_mrr_count=computed_program_mrr_count,
            kc_map_source=(
                "runtime_sample_point_line_global"
                if source_origin == "runtime_identified_profile"
                else "sample_point_line_global"
            ),
        )
        return sample_df

    def _store_manual_measurement_prediction(self, sample_df):
        """将样本级预测负载写回手动导入的实验实测数据。"""
        measurement = getattr(self, "manual_measurement_data", None)
        if not measurement:
            return
        for key in self._SEGMENTATION_PREDICTION_PROVENANCE_KEYS:
            measurement.pop(key, None)

        sample_df = self._initialize_measurement_prediction_channels(sample_df)
        sample_df = self._sync_display_prediction_aliases(sample_df)
        measurement["mapped_ap"] = pd.to_numeric(sample_df["ap"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        measurement["mapped_ae"] = pd.to_numeric(sample_df["ae"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        measurement["mapped_feed"] = pd.to_numeric(sample_df["feed_speed"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        measurement["mapped_mrr"] = pd.to_numeric(sample_df["mrr"], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        measurement["predicted_idle_power"] = sample_df["idle_power"].to_numpy(dtype=float)
        display_kc = self._clip_nonnegative_numeric_array(
            pd.to_numeric(sample_df.get("display_predicted_kc"), errors='coerce').to_numpy(dtype=float)
        )
        display_load = pd.to_numeric(sample_df.get("display_predicted_load"), errors='coerce')
        display_load = display_load.fillna(pd.to_numeric(sample_df["idle_power"], errors='coerce').fillna(0.0))
        interval_summary_kc = self._clip_nonnegative_numeric_array(
            pd.to_numeric(sample_df.get("interval_summary_kc"), errors='coerce').to_numpy(dtype=float)
        )
        interval_summary_load = pd.to_numeric(sample_df.get("interval_summary_load"), errors='coerce')
        interval_summary_load = interval_summary_load.fillna(pd.to_numeric(sample_df["idle_power"], errors='coerce').fillna(0.0))
        display_source = (
            sample_df["display_prediction_source"].astype(str).to_numpy(dtype=object)
            if "display_prediction_source" in sample_df.columns
            else sample_df["predicted_kc_source"].astype(str).to_numpy(dtype=object)
        )
        interval_summary_source = (
            sample_df["interval_summary_source"].astype(str).to_numpy(dtype=object)
            if "interval_summary_source" in sample_df.columns else display_source.copy()
        )
        measurement["mapped_kc"] = display_kc
        measurement["display_predicted_kc"] = display_kc
        measurement["interval_summary_kc"] = interval_summary_kc
        predicted_load = pd.to_numeric(sample_df["predicted_load"], errors='coerce')
        predicted_load = predicted_load.fillna(pd.to_numeric(sample_df["idle_power"], errors='coerce').fillna(0.0))
        measurement["predicted_load"] = np.maximum(predicted_load.to_numpy(dtype=float), 0.0)
        measurement["display_predicted_load"] = np.maximum(display_load.to_numpy(dtype=float), 0.0)
        measurement["interval_summary_load"] = np.maximum(interval_summary_load.to_numpy(dtype=float), 0.0)
        measurement["display_prediction_source"] = display_source
        measurement["interval_summary_source"] = interval_summary_source
        measurement["line_no_aligned"] = sample_df["line_no_aligned"].to_numpy(dtype=int)
        measurement["process_point_index"] = sample_df["process_point_index"].to_numpy(dtype=int)
        measurement["process_point_count"] = sample_df["process_point_count"].to_numpy(dtype=int)
        measurement["process_row_index"] = sample_df["process_row_index"].to_numpy(dtype=int)
        measurement["process_point_anchor_x"] = pd.to_numeric(
            sample_df.get("process_point_anchor_x"),
            errors="coerce",
        ).to_numpy(dtype=float)
        measurement["sample_anchor_x"] = pd.to_numeric(
            sample_df.get("sample_anchor_x"),
            errors="coerce",
        ).to_numpy(dtype=float)
        measurement["prediction_valid_mask"] = sample_df["prediction_valid"].to_numpy(dtype=bool)
        measurement["cutting_load"] = sample_df["cutting_load"].to_numpy(dtype=float)
        measurement["kc_point"] = sample_df["kc_point"].to_numpy(dtype=float)
        measurement["kc_valid_mask"] = sample_df["kc_valid"].to_numpy(dtype=bool)
        measurement["kc_gated_out_mask"] = sample_df["kc_gated_out"].to_numpy(dtype=bool)
        measurement["sample_kc_values"] = self._clip_nonnegative_numeric_array(
            pd.to_numeric(sample_df.get("sample_kc"), errors='coerce').to_numpy(dtype=float)
        )
        measurement["sample_kc_valid_mask"] = (
            sample_df["sample_kc_valid"].to_numpy(dtype=bool)
            if "sample_kc_valid" in sample_df.columns else np.isfinite(measurement["sample_kc_values"])
        )
        measurement["sample_kc_source"] = (
            sample_df["sample_kc_source"].to_numpy(dtype=object)
            if "sample_kc_source" in sample_df.columns
            else np.where(measurement["sample_kc_valid_mask"], "sample_direct", "").astype(object)
        )
        measurement["idle_window_mask"] = sample_df["idle_window"].to_numpy(dtype=bool)
        measurement["idle_point_mask"] = sample_df["is_idle_point"].to_numpy(dtype=bool)
        measurement["process_anchor_mask"] = (
            sample_df["process_anchor_mask"].to_numpy(dtype=bool)
            if "process_anchor_mask" in sample_df.columns else np.zeros(len(sample_df), dtype=bool)
        )
        measurement["sigma_idle"] = float(sample_df["sigma_idle"].iloc[0]) if not sample_df.empty else 0.0
        measurement["delta_mrr"] = float(sample_df["delta_mrr"].iloc[0]) if not sample_df.empty else 0.0
        measurement["display_mode"] = str(
            sample_df["display_mode"].iloc[0]
            if "display_mode" in sample_df.columns and not sample_df.empty
            else self._get_measurement_display_mode()
        )
        measurement["parameter_source"] = str(
            sample_df["parameter_source"].iloc[0]
            if "parameter_source" in sample_df.columns and not sample_df.empty
            else self._resolve_measurement_parameter_source()
        )
        measurement["interval_source"] = str(
            sample_df["interval_source"].iloc[0]
            if "interval_source" in sample_df.columns and not sample_df.empty
            else self._resolve_measurement_interval_source(fallback="measurement_runtime")
        )
        measurement["prediction_source"] = str(self._get_prediction_source())
        measurement["measurement_binding"] = self._build_manual_measurement_binding(measurement)
        measurement["measurement_runtime"] = {
            "updated_at": datetime.now().isoformat(),
            "sample_count": int(len(sample_df)),
        }
        if "gate_reference_kc" not in measurement:
            measurement["gate_reference_kc"] = float(self._resolve_measurement_gate_reference_kc())
        measurement["prediction_updated_at"] = datetime.now().isoformat()

    def _commit_current_program_idle_power(self, force=False):
        """在成功辨识后提交全局 P_idle；锁定时保留已有全局值。"""
        try:
            candidate_idle = float(self.current_program_idle_power.get())
        except Exception:
            candidate_idle = float(self.p_idle_var.get() or 0.0)

        if not np.isfinite(candidate_idle) or candidate_idle <= 0.0:
            try:
                candidate_idle = float(self.p_idle_var.get())
            except Exception:
                candidate_idle = 0.0

        if not np.isfinite(candidate_idle):
            candidate_idle = 0.0

        locked_idle = None if force else self._resolve_fixed_global_idle_power()
        idle_locked = locked_idle is not None
        committed_idle = float(locked_idle if idle_locked else candidate_idle)

        if not idle_locked:
            self.p_idle_var.set(float(committed_idle))
        if hasattr(self, "_update_program_idle_summary"):
            self._update_program_idle_summary()
        return {
            "candidate_idle": float(candidate_idle),
            "committed_idle": float(committed_idle),
            "idle_locked": bool(idle_locked),
        }

    def _apply_current_interval_mode_kc_override_to_measurement(self):
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            return None
        if not getattr(self, "manual_measurement_data", None):
            return None
        if self._is_imported_profile_forward_lock_active():
            return None
        if not self._get_current_interval_records(allow_profile_fallback=False):
            return None
        return self._refresh_manual_measurement_prediction(
            allow_saved_sample_profile=False,
            allow_measurement_resolve=True,
            display_mode="posterior",
        )

    def _refresh_manual_measurement_prediction(
        self,
        allow_saved_sample_profile=True,
        allow_measurement_resolve=None,
        display_mode=None,
    ):
        """使用当前 P_idle/K_c/K_e 刷新实验实测的样本级预测负载。"""
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            return None
        if not getattr(self, "manual_measurement_data", None):
            return None
        if not self.data:
            return None

        measurement = getattr(self, "manual_measurement_data", None)
        measurement_case_signature = self._get_current_measurement_case_signature(measurement)
        self.measurement_case_signature = measurement_case_signature
        allow_imported_autoload = self._should_allow_imported_profile_autoload(
            allow_measurement_resolve=allow_measurement_resolve,
        )
        profile_origin, forward_profile = self._resolve_forward_prediction_profile(
            measurement=measurement,
            process_path=self._get_primary_input_file_or_empty(),
            allow_autoload_imported=allow_imported_autoload,
        )
        if profile_origin == "imported_profile" and isinstance(forward_profile, dict):
            display_mode = "forward"
            allow_measurement_resolve = False
        effective_display_mode = str(
            display_mode or self._get_measurement_display_mode(prediction_source=profile_origin)
        ).strip() or "forward"
        reverse_solve = self._should_reverse_solve_measurement_prediction(
            allow_measurement_resolve=allow_measurement_resolve,
            profile_origin=profile_origin,
            profile=forward_profile,
            allow_autoload_imported=allow_imported_autoload,
            display_mode=effective_display_mode,
        )
        if not reverse_solve and not isinstance(forward_profile, dict):
            reverse_solve = True
            profile_origin = "no_profile"
        if reverse_solve:
            self._resolve_measurement_gate_reference_kc()
        sample_df = self._build_manual_measurement_sample_frame(
            allow_saved_sample_profile=bool(allow_saved_sample_profile),
        )
        interval_records = []
        interval_source = "none"
        if reverse_solve:
            sigma_idle, delta_mrr, idle_count, idle_mask = self._estimate_idle_sigma_and_delta_mrr(
                sample_df,
                kc_reference=self._resolve_measurement_gate_reference_kc(),
            )
            sample_df = self._append_manual_measurement_impedance(sample_df, sigma_idle, delta_mrr, idle_mask)
            interval_records = self._get_current_interval_records(allow_profile_fallback=False)
            interval_source = self._resolve_measurement_interval_source(
                interval_records,
                fallback="measurement_point_kc",
            )
            if interval_records:
                sample_df = self._apply_steady_interval_kc_to_sample_df(
                    sample_df,
                    intervals=interval_records,
                    authoritative_interval_kc=False,
                    write_to_prediction=True,
                )
            self._update_measurement_gate_indicators(sigma_idle, delta_mrr, idle_count)
        else:
            sample_df = self._apply_profile_forward_prediction_to_sample_df(
                sample_df,
                profile=forward_profile,
                source_label=profile_origin,
            )
            interval_source = profile_origin if profile_origin == "imported_profile" else "runtime_interval_summary"
            if profile_origin == "runtime_identified_profile":
                sigma_idle, delta_mrr, idle_count, idle_mask = self._estimate_idle_sigma_and_delta_mrr(
                    sample_df,
                    kc_reference=self._resolve_measurement_gate_reference_kc(),
                )
                sample_df = self._append_manual_measurement_impedance(sample_df, sigma_idle, delta_mrr, idle_mask)
                interval_records = self._resolve_interval_records_for_measurement_prediction()
                interval_source = self._resolve_measurement_interval_source(
                    interval_records,
                    fallback="runtime_interval_summary",
                )
                if interval_records:
                    sample_df = self._apply_steady_interval_kc_to_sample_df(
                        sample_df,
                        intervals=interval_records,
                        authoritative_interval_kc=False,
                        write_to_prediction=True,
                    )
                self._update_measurement_gate_indicators(sigma_idle, delta_mrr, idle_count)
            else:
                self._update_measurement_gate_indicators(0.0, 0.0, 0)
        sample_df = self._sync_display_prediction_aliases(sample_df)
        sample_df["display_mode"] = effective_display_mode
        sample_df["parameter_source"] = self._resolve_measurement_parameter_source(prediction_source=profile_origin)
        sample_df["interval_source"] = str(interval_source or "none")
        sample_df["prediction_source"] = str(self._get_prediction_source())
        self._store_manual_measurement_prediction(sample_df)
        if reverse_solve or profile_origin == "runtime_identified_profile":
            segmentation_prediction_source = "measurement_reverse"
            segmentation_prediction_independent = False
        elif profile_origin == "imported_profile":
            independence_checker = getattr(
                self,
                "_profile_is_independent_from_current_measurement",
                None,
            )
            profile_is_independent = bool(
                isinstance(forward_profile, dict)
                and callable(independence_checker)
                and independence_checker(
                    forward_profile,
                    measurement=measurement,
                )
            )
            segmentation_prediction_source = (
                "independent_profile"
                if profile_is_independent
                else "same_measurement_profile"
            )
            segmentation_prediction_independent = bool(profile_is_independent)
        else:
            segmentation_prediction_source = "unclassified"
            segmentation_prediction_independent = False
        measurement["segmentation_prediction_source"] = str(
            segmentation_prediction_source
        )
        measurement["segmentation_prediction_independent"] = bool(
            segmentation_prediction_independent
        )
        measurement["segmentation_temporary_measurement_mode"] = bool(
            segmentation_prediction_source
            in {"measurement_reverse", "same_measurement_profile"}
        )
        measurement["segmentation_sample_prediction_context_signature"] = (
            self._build_prediction_context_signature(
                prediction_source=profile_origin,
                measurement=measurement,
            )
        )
        self._debug_prediction_state_event(
            "refresh_manual_measurement_prediction",
            measurement_case_signature=measurement_case_signature or "none",
            reverse_solve=bool(reverse_solve),
            parameter_source=self._resolve_measurement_parameter_source(prediction_source=profile_origin),
            display_mode=effective_display_mode,
            interval_source=str(interval_source or "none"),
            live_display=effective_display_mode,
            sample_prediction_context_signature=measurement[
                "segmentation_sample_prediction_context_signature"
            ],
            kc_map_source=(
                "imported_profile"
                if profile_origin == "imported_profile"
                else ("runtime_fit" if profile_origin == "runtime_identified_profile" else "current_rows")
            ),
        )
        return sample_df

    def _build_identification_line_groups(self, sample_df):
        valid_mask = (
            sample_df["prediction_valid"].to_numpy(dtype=bool)
            & np.isfinite(sample_df["actual_load"].to_numpy(dtype=float))
            & np.isfinite(sample_df["cutting_load"].to_numpy(dtype=float))
            & (sample_df["cutting_load"].to_numpy(dtype=float) > 0.0)
            & np.isfinite(sample_df["mrr"].to_numpy(dtype=float))
            & (sample_df["mrr"].to_numpy(dtype=float) > 1e-9)
            & (sample_df["ap"].to_numpy(dtype=float) > 0.0)
            & (sample_df["ae"].to_numpy(dtype=float) > 0.0)
        )
        working_df = sample_df.loc[valid_mask].copy()
        if working_df.empty:
            raise ValueError("当前实验实测与工艺信息表对齐后，没有可用于辨识的有效切削样本")

        working_df = working_df.reset_index(drop=True)
        sample_indices = working_df["sample_index"].to_numpy(dtype=int)
        line_values = pd.to_numeric(working_df["line_no_raw"], errors='coerce').fillna(-1).to_numpy(dtype=int)
        if "process_point_index" in working_df.columns:
            point_series = working_df["process_point_index"]
        else:
            point_series = pd.Series(np.full(len(working_df), -1, dtype=int), index=working_df.index)
        point_values = pd.to_numeric(point_series, errors='coerce').fillna(-1).to_numpy(dtype=int)
        load_values_all = working_df["actual_load"].to_numpy(dtype=float)
        spindle_values_all = working_df["spindle_speed"].to_numpy(dtype=float)
        feed_values_all = working_df["feed_speed"].to_numpy(dtype=float)
        mrr_values_all = working_df["mrr"].to_numpy(dtype=float)
        cutting_values_all = working_df["cutting_load"].to_numpy(dtype=float)
        ap_values_all = working_df["ap"].to_numpy(dtype=float)
        ae_values_all = working_df["ae"].to_numpy(dtype=float)
        idle_values_all = pd.to_numeric(working_df["idle_power"], errors='coerce').fillna(0.0).to_numpy(dtype=float)

        min_points = 30
        mrr_tol = 0.08
        ap_tol = 0.05
        ae_tol = 0.05
        load_tol = 0.18
        speed_tol = 0.02
        records = []

        def _relative_span_np(values):
            finite = np.asarray(values, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                return float("nan")
            center = float(np.median(finite))
            low = float(np.percentile(finite, 5))
            high = float(np.percentile(finite, 95))
            scale = max(abs(center), 1.0)
            return abs(high - low) / scale

        def _segment_within_tolerance(segment_record):
            return (
                np.isfinite(segment_record["mrr_rel_span"])
                and np.isfinite(segment_record["ap_rel_span"])
                and np.isfinite(segment_record["ae_rel_span"])
                and np.isfinite(segment_record["load_rel_span"])
                and np.isfinite(segment_record["speed_rel_span"])
                and segment_record["mrr_rel_span"] <= mrr_tol
                and segment_record["ap_rel_span"] <= ap_tol
                and segment_record["ae_rel_span"] <= ae_tol
                and segment_record["load_rel_span"] <= load_tol
                and segment_record["speed_rel_span"] <= speed_tol
                and np.isfinite(segment_record["cutting_load"])
                and segment_record["cutting_load"] > 0.0
            )

        def _segment_pass(segment_record):
            return (
                segment_record["sample_count"] >= min_points
                and _segment_within_tolerance(segment_record)
            )

        def _build_record(seg_start, seg_end, line_group_count):
            start_idx = int(seg_start)
            end_idx = int(seg_end)
            if end_idx < start_idx:
                return None

            seg_slice = slice(start_idx, end_idx + 1)
            record = {
                "start_pos": start_idx,
                "end_pos": end_idx,
                "start_sample_index": int(sample_indices[start_idx]),
                "end_sample_index": int(sample_indices[end_idx]),
                "start_line": int(line_values[start_idx]),
                "end_line": int(line_values[end_idx]),
                "line_group_count": int(line_group_count),
                "sample_count": int(end_idx - start_idx + 1),
                "ap": float(np.median(ap_values_all[seg_slice])),
                "ae": float(np.median(ae_values_all[seg_slice])),
                "feed_speed": float(np.median(feed_values_all[seg_slice])),
                "spindle_speed": float(np.median(spindle_values_all[seg_slice])),
                "actual_load": float(np.median(load_values_all[seg_slice])),
                "idle_power": float(np.median(idle_values_all[seg_slice])),
                "mrr": float(np.median(mrr_values_all[seg_slice])),
                "cutting_load": float(np.median(cutting_values_all[seg_slice])),
                "mrr_rel_span": _relative_span_np(mrr_values_all[seg_slice]),
                "load_rel_span": _relative_span_np(load_values_all[seg_slice]),
                "speed_rel_span": _relative_span_np(spindle_values_all[seg_slice]),
                "feed_rel_span": _relative_span_np(feed_values_all[seg_slice]),
                "ap_rel_span": _relative_span_np(ap_values_all[seg_slice]),
                "ae_rel_span": _relative_span_np(ae_values_all[seg_slice]),
            }
            record["steady_pass"] = _segment_pass(record)
            return record

        def _flush_segment(seg_start, seg_end, line_group_count):
            record = _build_record(seg_start, seg_end, line_group_count)
            if record is not None:
                records.append(record)

        run_records = []
        run_start = 0
        for idx in range(1, len(working_df) + 1):
            end_of_data = idx >= len(working_df)
            same_run = False
            if not end_of_data:
                same_run = (
                    sample_indices[idx] == sample_indices[idx - 1] + 1
                    and line_values[idx] == line_values[idx - 1]
                    and point_values[idx] == point_values[idx - 1]
                )
            if same_run:
                continue
            run_record = _build_record(run_start, idx - 1, 1)
            if run_record is not None:
                run_records.append(run_record)
            run_start = idx

        if not run_records:
            raise ValueError("未能从实验实测中提取到候选 MRR 稳态区间")

        def _init_stats(record):
            return {
                "sample_count": int(record["sample_count"]),
                "group_count": 1,
                "mrr_min": float(record["mrr"]),
                "mrr_max": float(record["mrr"]),
                "ap_min": float(record["ap"]),
                "ap_max": float(record["ap"]),
                "ae_min": float(record["ae"]),
                "ae_max": float(record["ae"]),
                "load_min": float(record["actual_load"]),
                "load_max": float(record["actual_load"]),
                "speed_min": float(record["spindle_speed"]),
                "speed_max": float(record["spindle_speed"]),
                "feed_min": float(record["feed_speed"]),
                "feed_max": float(record["feed_speed"]),
            }

        def _extend_stats(stats, record):
            return {
                "sample_count": int(stats["sample_count"]) + int(record["sample_count"]),
                "group_count": int(stats["group_count"]) + 1,
                "mrr_min": min(float(stats["mrr_min"]), float(record["mrr"])),
                "mrr_max": max(float(stats["mrr_max"]), float(record["mrr"])),
                "ap_min": min(float(stats["ap_min"]), float(record["ap"])),
                "ap_max": max(float(stats["ap_max"]), float(record["ap"])),
                "ae_min": min(float(stats["ae_min"]), float(record["ae"])),
                "ae_max": max(float(stats["ae_max"]), float(record["ae"])),
                "load_min": min(float(stats["load_min"]), float(record["actual_load"])),
                "load_max": max(float(stats["load_max"]), float(record["actual_load"])),
                "speed_min": min(float(stats["speed_min"]), float(record["spindle_speed"])),
                "speed_max": max(float(stats["speed_max"]), float(record["spindle_speed"])),
                "feed_min": min(float(stats["feed_min"]), float(record["feed_speed"])),
                "feed_max": max(float(stats["feed_max"]), float(record["feed_speed"])),
            }

        def _segment_within_stats(stats):
            candidate = {
                "sample_count": int(stats["sample_count"]),
                "mrr_rel_span": self._relative_span_from_bounds(stats["mrr_min"], stats["mrr_max"]),
                "ap_rel_span": self._relative_span_from_bounds(stats["ap_min"], stats["ap_max"]),
                "ae_rel_span": self._relative_span_from_bounds(stats["ae_min"], stats["ae_max"]),
                "load_rel_span": self._relative_span_from_bounds(stats["load_min"], stats["load_max"]),
                "speed_rel_span": self._relative_span_from_bounds(stats["speed_min"], stats["speed_max"]),
                "feed_rel_span": self._relative_span_from_bounds(stats["feed_min"], stats["feed_max"]),
                "cutting_load": max(float(stats["load_min"]), float(stats["load_max"])),
            }
            return _segment_within_tolerance(candidate)

        seg_start_idx = 0
        seg_stats = _init_stats(run_records[0])
        for run_idx in range(1, len(run_records)):
            prev_record = run_records[run_idx - 1]
            current_record = run_records[run_idx]
            contiguous = int(current_record["start_sample_index"]) == int(prev_record["end_sample_index"]) + 1
            candidate_stats = _extend_stats(seg_stats, current_record)
            if (not contiguous) or not _segment_within_stats(candidate_stats):
                first_record = run_records[seg_start_idx]
                last_record = run_records[run_idx - 1]
                _flush_segment(
                    int(first_record["start_pos"]),
                    int(last_record["end_pos"]),
                    int(seg_stats["group_count"]),
                )
                seg_start_idx = run_idx
                seg_stats = _init_stats(current_record)
            else:
                seg_stats = candidate_stats

        first_record = run_records[seg_start_idx]
        last_record = run_records[-1]
        _flush_segment(
            int(first_record["start_pos"]),
            int(last_record["end_pos"]),
            int(seg_stats["group_count"]),
        )

        line_group_df = pd.DataFrame.from_records(records)
        if line_group_df.empty:
            raise ValueError("未能从实验实测中提取到候选 MRR 稳态区间")
        return working_df, line_group_df

    def _summarize_identification_interval(self, sample_df, start_pos, end_pos, line_group_count):
        segment = sample_df.iloc[int(start_pos):int(end_pos) + 1]
        if segment.empty:
            return None

        load_values = segment["actual_load"].to_numpy(dtype=float)
        spindle_values = segment["spindle_speed"].to_numpy(dtype=float)
        feed_values = segment["feed_speed"].to_numpy(dtype=float)
        mrr_values = segment["mrr"].to_numpy(dtype=float)
        cutting_values = segment["cutting_load"].to_numpy(dtype=float)
        ap_values = segment["ap"].to_numpy(dtype=float)
        ae_values = segment["ae"].to_numpy(dtype=float)

        return {
            "start_pos": int(start_pos),
            "end_pos": int(end_pos),
            "start_sample_index": int(segment["sample_index"].iloc[0]),
            "end_sample_index": int(segment["sample_index"].iloc[-1]),
            "start_line": int(segment["line_no_raw"].iloc[0]),
            "end_line": int(segment["line_no_raw"].iloc[-1]),
            "line_group_count": int(line_group_count),
            "sample_count": int(len(segment)),
            "ap": float(np.median(ap_values)),
            "ae": float(np.median(ae_values)),
            "feed_speed": float(np.median(feed_values)),
            "spindle_speed": float(np.median(spindle_values)),
            "actual_load": float(np.median(load_values)),
            "idle_power": float(np.median(segment["idle_power"].to_numpy(dtype=float))),
            "mrr": float(np.median(mrr_values)),
            "cutting_load": float(np.median(cutting_values)),
            "mrr_rel_span": self._relative_span(mrr_values),
            "load_rel_span": self._relative_span(load_values),
            "speed_rel_span": self._relative_span(spindle_values),
            "feed_rel_span": self._relative_span(feed_values),
        }

    def _merge_identification_intervals(self, sample_df, line_group_df):
        interval_df = line_group_df.loc[line_group_df["steady_pass"]].copy()
        if interval_df.empty:
            raise ValueError("未找到满足 MRR 稳定判据的候选区间")
        return interval_df.sort_values("start_pos").reset_index(drop=True)

    def _select_identification_geometry_group(self, interval_df):
        working_df = interval_df.copy()
        ap_values = pd.to_numeric(working_df["ap"], errors='coerce').to_numpy(dtype=float)
        mrr_values = pd.to_numeric(working_df["mrr"], errors='coerce').to_numpy(dtype=float)
        cutting_values = pd.to_numeric(working_df["cutting_load"], errors='coerce').to_numpy(dtype=float)
        specific_mrr = np.divide(
            mrr_values,
            ap_values,
            out=np.full_like(mrr_values, np.nan, dtype=float),
            where=np.abs(ap_values) > 1e-12,
        )
        normalized_load = np.divide(
            cutting_values,
            ap_values,
            out=np.full_like(cutting_values, np.nan, dtype=float),
            where=np.abs(ap_values) > 1e-12,
        )
        working_df["specific_mrr"] = specific_mrr
        working_df["normalized_load"] = normalized_load

        valid_mask = (
            np.isfinite(ap_values)
            & np.isfinite(mrr_values)
            & np.isfinite(cutting_values)
            & np.isfinite(specific_mrr)
            & np.isfinite(normalized_load)
            & (ap_values > 1e-12)
            & (mrr_values > 1e-12)
            & (cutting_values > 0.0)
        )
        working_df = working_df.loc[valid_mask].sort_values("start_pos").reset_index(drop=True)
        if len(working_df) < 2:
            raise ValueError("近恒定 ap/ae/F 代表区不足 2 段，无法辨识 K_c / K_e")

        return working_df

    def _fit_kc_ke_from_intervals(self, interval_df, fixed_ke=None):
        return self._fit_kc_ke_by_linear_system(
            interval_df["mrr"],
            interval_df["ap"],
            interval_df["cutting_load"],
            "候选代表区",
            fixed_ke=fixed_ke,
        )

    def _fit_kc_ke_from_general_points(self, point_records, fixed_ke=None):
        point_df = pd.DataFrame.from_records(point_records or [])
        if point_df.empty:
            raise ValueError("未选中可用于反算的点")

        mrr_values = pd.to_numeric(point_df["mrr"], errors='coerce').to_numpy(dtype=float)
        ap_values = pd.to_numeric(point_df["ap"], errors='coerce').to_numpy(dtype=float)
        y_values = pd.to_numeric(point_df["cutting_load"], errors='coerce').to_numpy(dtype=float)
        finite_mask = (
            np.isfinite(mrr_values)
            & np.isfinite(ap_values)
            & np.isfinite(y_values)
            & (mrr_values > 1e-12)
            & (ap_values > 1e-12)
        )
        mrr_values = mrr_values[finite_mask]
        ap_values = ap_values[finite_mask]
        y_values = y_values[finite_mask]

        if len(mrr_values) < 2:
            target_label = "K_c" if fixed_ke is not None else "K_c / K_e"
            raise ValueError(f"已选点不足 2 个，无法反算 {target_label}")
        return self._fit_kc_ke_by_linear_system(
            mrr_values,
            ap_values,
            y_values,
            "已选点",
            fixed_ke=fixed_ke,
        )

    def _build_measurement_identification_fit_frame(self, sample_df):
        if sample_df is None or sample_df.empty:
            raise ValueError("未生成可用于辨识的样本级预测数据")

        working_df = sample_df.copy()
        mrr_values = pd.to_numeric(working_df["mrr"], errors='coerce').to_numpy(dtype=float)
        ap_values = pd.to_numeric(working_df["ap"], errors='coerce').to_numpy(dtype=float)
        ae_values = pd.to_numeric(working_df["ae"], errors='coerce').to_numpy(dtype=float)
        feed_values = pd.to_numeric(working_df["feed_speed"], errors='coerce').to_numpy(dtype=float)
        actual_load = pd.to_numeric(working_df["actual_load"], errors='coerce').to_numpy(dtype=float)
        idle_power = pd.to_numeric(working_df["idle_power"], errors='coerce').to_numpy(dtype=float)
        cutting_load = actual_load - idle_power
        idle_point_mask = np.asarray(
            working_df["is_idle_point"] if "is_idle_point" in working_df.columns else np.zeros(len(working_df), dtype=bool),
            dtype=bool,
        )
        transition_spike_mask = np.asarray(
            working_df["transition_spike"] if "transition_spike" in working_df.columns else np.zeros(len(working_df), dtype=bool),
            dtype=bool,
        )
        anchor_mask = np.asarray(
            working_df["process_anchor_mask"] if "process_anchor_mask" in working_df.columns else np.ones(len(working_df), dtype=bool),
            dtype=bool,
        )

        if "delta_mrr" in working_df.columns and len(working_df):
            try:
                delta_mrr = float(pd.to_numeric(working_df["delta_mrr"], errors="coerce").iloc[0])
            except Exception:
                delta_mrr = float("nan")
        else:
            delta_mrr = float("nan")
        if not np.isfinite(delta_mrr):
            delta_mrr = float(getattr(self, "manual_measurement_data", {}).get("delta_mrr", 0.0) or 0.0)
        valid_mask = (
            working_df["prediction_valid"].to_numpy(dtype=bool)
            & np.isfinite(cutting_load)
            & np.isfinite(mrr_values)
            & np.isfinite(ap_values)
            & np.isfinite(ae_values)
            & np.isfinite(feed_values)
            & (cutting_load > 0.0)
            & (mrr_values > max(delta_mrr, 1e-12))
            & (ap_values > 1e-12)
            & (ae_values > 1e-12)
            & anchor_mask
            & ~idle_point_mask
            & ~transition_spike_mask
        )

        fit_columns = [
            "sample_index",
            "line_no_raw",
            "line_no_aligned",
            "process_point_index",
            "process_point_count",
            "ap",
            "ae",
            "feed_speed",
            "spindle_speed",
            "mrr",
            "actual_load",
            "idle_power",
        ]
        available_columns = [name for name in fit_columns if name in working_df.columns]
        fit_df = working_df.loc[valid_mask, available_columns].copy()
        if fit_df.empty:
            raise ValueError("去除空载点后，没有可用于按点辨识的切削样本")

        fit_df["cutting_load"] = cutting_load[valid_mask]
        fit_df["specific_mrr"] = np.divide(
            fit_df["mrr"].to_numpy(dtype=float),
            fit_df["ap"].to_numpy(dtype=float),
            out=np.full(len(fit_df), np.nan, dtype=float),
            where=np.abs(fit_df["ap"].to_numpy(dtype=float)) > 1e-12,
        )
        return fit_df

    def _update_manual_kcke_button_text(self):
        button = getattr(self, "manual_kcke_btn", None)
        if button is None:
            return
        target_label = self._get_identification_short_label()
        if self.manual_kcke_pick_mode:
            count = len(self.manual_kcke_points or [])
            suffix = f" ({count})" if count else ""
            button.configure(text=f"✅ 计算{target_label}{suffix}")
        else:
            button.configure(text=f"📍 点选{target_label}")

    def _clear_manual_kcke_markers(self, clear_points=False, redraw=True):
        for artist in list(getattr(self, "manual_kcke_marker_artists", [])):
            try:
                artist.remove()
            except Exception:
                pass
        self.manual_kcke_marker_artists = []
        if clear_points:
            self.manual_kcke_points = []
        if redraw and getattr(self, "canvas_data", None) is not None:
            try:
                self.canvas_data.draw_idle()
            except Exception:
                pass

    def _restore_manual_kcke_markers(self):
        self._clear_manual_kcke_markers(clear_points=False, redraw=False)
        if not getattr(self, "manual_kcke_points", None):
            return
        fig = getattr(self, "_current_preview_fig", None)
        canvas = getattr(self, "canvas_data", None)
        if fig is None or not getattr(fig, "axes", None):
            return

        base_ax = fig.axes[0]
        y_top = base_ax.get_ylim()[1]
        for idx, record in enumerate(self.manual_kcke_points, start=1):
            x_value = record.get("x_display")
            if x_value is None or not np.isfinite(float(x_value)):
                continue
            for ax in fig.axes:
                try:
                    line = ax.axvline(float(x_value), color="#FF8C00", linestyle="--", linewidth=1.0, alpha=0.9, zorder=30)
                    self.manual_kcke_marker_artists.append(line)
                except Exception:
                    continue
            try:
                text = base_ax.text(
                    float(x_value), y_top, f"P{idx}",
                    color="#FF8C00", fontsize=9, ha="center", va="bottom", zorder=31
                )
                self.manual_kcke_marker_artists.append(text)
            except Exception:
                pass
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass

    def _summarize_manual_kcke_pick(self, x_value):
        measurement = getattr(self, "manual_measurement_data", None)
        display_x_all = getattr(self, "_current_sample_display_x", None)
        context_mask = getattr(self, "_current_sample_context_mask", None)
        valid_mask = getattr(self, "_current_sample_valid_mask", None)
        if measurement is None or display_x_all is None or context_mask is None:
            raise ValueError("当前预览中没有可点选的实验实测数据")

        if valid_mask is not None and np.any(valid_mask):
            pick_mask = np.asarray(valid_mask, dtype=bool)
        else:
            pick_mask = np.asarray(context_mask, dtype=bool)
        finite_x_mask = np.isfinite(display_x_all)
        pick_mask = pick_mask & finite_x_mask

        if "predicted_idle_power" not in measurement or "mapped_mrr" not in measurement:
            self._refresh_manual_measurement_prediction()
            measurement = getattr(self, "manual_measurement_data", None)
            if measurement is None:
                raise ValueError("样本级映射尚未生成")

        line_values = np.asarray(measurement.get("program_line", []), dtype=int)
        actual_load = np.asarray(measurement.get("actual_load", []), dtype=float)
        idle_power = np.asarray(measurement.get("predicted_idle_power", []), dtype=float)
        ap_values = np.asarray(measurement.get("mapped_ap", []), dtype=float)
        ae_values = np.asarray(measurement.get("mapped_ae", []), dtype=float)
        mrr_values = np.asarray(measurement.get("mapped_mrr", []), dtype=float)
        feed_values = np.asarray(measurement.get("mapped_feed", []), dtype=float)
        speed_values = np.asarray(measurement.get("actual_spindle_speed", []), dtype=float)
        aligned_values = np.asarray(measurement.get("line_no_aligned", line_values), dtype=int)

        geometry_mask = (
            np.isfinite(ap_values)
            & np.isfinite(ae_values)
            & np.isfinite(mrr_values)
            & (ap_values > 1e-12)
            & (ae_values > 1e-12)
            & (mrr_values > 1e-12)
        )
        pick_mask = pick_mask & geometry_mask
        pick_indices = np.flatnonzero(pick_mask)
        if pick_indices.size == 0:
            raise ValueError("当前视图没有可点选的有效样本（切深/切宽/MRR 必须大于 0）")

        nearest_local = int(np.argmin(np.abs(display_x_all[pick_indices] - float(x_value))))
        center_idx = int(pick_indices[nearest_local])

        if center_idx >= len(line_values):
            raise ValueError("点选位置超出样本范围")

        line_value = int(line_values[center_idx])
        left = center_idx
        right = center_idx
        while left - 1 >= 0 and pick_mask[left - 1] and int(line_values[left - 1]) == line_value:
            left -= 1
        while right + 1 < len(line_values) and pick_mask[right + 1] and int(line_values[right + 1]) == line_value:
            right += 1

        sample_slice = slice(left, right + 1)
        ap_med = float(np.nanmedian(ap_values[sample_slice]))
        ae_med = float(np.nanmedian(ae_values[sample_slice]))
        mrr_med = float(np.nanmedian(mrr_values[sample_slice]))
        if not np.isfinite(ap_med) or not np.isfinite(ae_med) or ap_med <= 0 or ae_med <= 0:
            raise ValueError(f"所选点所在程序行 {line_value + 1} 没有可用的切深/切宽")
        if not np.isfinite(mrr_med) or mrr_med <= 1e-12:
            raise ValueError(f"所选点所在程序行 {line_value + 1} 的 MRR 无效")

        actual_segment = actual_load[sample_slice]
        idle_segment = idle_power[sample_slice]
        cutting_segment = actual_segment - idle_segment
        actual_var = float(np.nanvar(actual_segment, ddof=1)) if len(actual_segment) > 1 else 0.0
        actual_std = float(math.sqrt(max(actual_var, 0.0)))

        return {
            "line_no_raw": line_value,
            "line_no_aligned": int(np.nanmedian(aligned_values[sample_slice])) if len(aligned_values[sample_slice]) else line_value,
            "start_idx": int(left),
            "end_idx": int(right),
            "sample_count": int(right - left + 1),
            "ap": ap_med,
            "ae": ae_med,
            "mrr": mrr_med,
            "specific_mrr": float(mrr_med / ap_med) if abs(ap_med) > 1e-12 else float("nan"),
            "actual_load": float(np.nanmedian(actual_segment)),
            "idle_power": float(np.nanmedian(idle_segment)),
            "cutting_load": float(np.nanmedian(cutting_segment)),
            "feed_speed": float(np.nanmedian(feed_values[sample_slice])),
            "spindle_speed": float(np.nanmedian(speed_values[sample_slice])),
            "actual_load_std": actual_std,
            "actual_load_var": actual_var,
            "x_display": float(np.nanmedian(display_x_all[sample_slice])),
        }

    def on_manual_kcke_plot_click(self, event):
        if not getattr(self, "manual_kcke_pick_mode", False):
            return False
        if event is None or event.inaxes is None or event.xdata is None:
            return False

        try:
            record = self._summarize_manual_kcke_pick(event.xdata)
        except Exception as exc:
            self.set_status(f"点选失败: {str(exc)}", 4000)
            return True

        for existing in self.manual_kcke_points:
            if existing.get("start_idx") == record["start_idx"] and existing.get("end_idx") == record["end_idx"]:
                self.set_status(f"程序行 {record['line_no_raw'] + 1} 已经选过", 3000)
                return True

        self.manual_kcke_points.append(record)
        self._update_manual_kcke_button_text()
        self._restore_manual_kcke_markers()
        self.set_status(
            f"已选 {len(self.manual_kcke_points)} 个点: 行号 {record['line_no_raw'] + 1}, "
            f"ap={record['ap']:.4f}, ae={record['ae']:.4f}, F={record['feed_speed']:.3f}, "
            f"ae·F/60={record['specific_mrr']:.3f}",
            4000
        )
        return True

    def toggle_manual_kcke_pick_mode(self):
        target_label = self._get_identification_target_label()
        short_label = self._get_identification_short_label()
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            messagebox.showwarning("不可用", f"点选{short_label}仅适用于实验实测模式")
            return
        if not getattr(self, "manual_measurement_data", None):
            messagebox.showwarning("不可用", "请先导入实验实测文件")
            return
        if not self.data:
            if not self._process_current_input_for_preview():
                messagebox.showwarning("不可用", "请先处理并显示当前工艺信息文件")
                return

        if not self.manual_kcke_pick_mode:
            self.manual_kcke_pick_mode = True
            self.manual_kcke_points = []
            self._clear_manual_kcke_markers(clear_points=False, redraw=False)
            self._update_manual_kcke_button_text()
            self.step_feed_status_var.set(
                f"点选模式已开启: 请优先选取 ap/ae 稳定且 F 有明显变化的稳态点，再点击按钮计算 {target_label}"
            )
            self.set_status("点选模式已开启，请优先选择 ap/ae 稳定且 F 有明显变化的稳态点", 5000)
            return

        if len(self.manual_kcke_points) < 2:
            self.set_status(f"至少点选 2 个稳态点后才能计算 {target_label}", 4000)
            return

        try:
            fit_result = self._fit_kc_ke_from_general_points(
                self.manual_kcke_points,
                fixed_ke=self._resolve_fixed_ke_for_identification(),
            )
        except Exception as exc:
            messagebox.showerror("计算失败", f"点选反算 {target_label} 失败:\n{str(exc)}")
            self.set_status(f"点选反算失败: {str(exc)}", 5000)
            return

        line_text = ",".join(str(int(item.get("line_no_raw", 0)) + 1) for item in self.manual_kcke_points)
        idle_commit_result = self._commit_current_program_idle_power()
        self._apply_cutting_model_fit_result(fit_result)
        self.step_feed_model_signature = (
            f"manual_points|{line_text}|{fit_result['identification_mode']}|"
            f"{fit_result['kc_value']:.6f}|{fit_result['ke_value']:.6f}|{fit_result['kc_sigma']:.6f}"
        )
        self._sync_prediction_mode_after_model_change(prefer_posterior=True)
        self.step_feed_status_var.set(
            f"已按点选反算({self._format_identification_mode_text(fit_result)}): 点数={len(self.manual_kcke_points)}, "
            f"{self._format_idle_commit_text(idle_commit_result)}, 全局K_c={fit_result['kc_value']:.6f}, "
            f"{self._format_identification_ke_text(fit_result)}, "
            f"σ_Kc={fit_result['kc_sigma']:.6f}, K_c^UCB={fit_result['kc_ucb']:.6f}"
        )
        self.manual_kcke_pick_mode = False
        self._update_manual_kcke_button_text()
        self._persist_app_config()
        self.set_status(f"点选反算 {target_label} 完成", 4000)
        if self.get_primary_input_file():
            self._force_recompute_kc_profile = True
            try:
                self._refresh_manual_measurement_prediction(
                    allow_saved_sample_profile=False,
                    allow_measurement_resolve=True,
                )
                self.generate_plots(
                    save=False,
                    silent=True,
                    interval_policy="reuse_current_template",
                    persist_profile=False,
                )
            finally:
                self._force_recompute_kc_profile = False
            self._debug_interval_state_event(
                "reidentify_overwrite_current_state",
                source="manual_points",
                interval_count=len(self._get_current_interval_records(allow_profile_fallback=False)),
                segment_count=len(self._get_current_segment_records(allow_profile_fallback=False)),
                profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
            )
            self._persist_current_kc_profile(source="manual_points")

    def summarize_measurement_interval(self, start_x, end_x, program_no=None):
        """按显示区间提取实验实测块，计算稳态门控和区间级 Kc 统计量。"""
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            return None
        measurement = getattr(self, "manual_measurement_data", None)
        if not measurement or self.sample_data_x_positions is None:
            return None

        blocks = self._get_sample_x_range_blocks(start_x, end_x, program_no=program_no)
        if not blocks:
            return None

        block_start, block_end = max(blocks, key=lambda item: item[1] - item[0] + 1)
        sample_slice = slice(int(block_start), int(block_end) + 1)
        actual_load_raw = np.asarray(measurement.get("actual_load", []), dtype=float)[sample_slice]
        if actual_load_raw.size == 0:
            return None

        actual_load = actual_load_raw[np.isfinite(actual_load_raw)]
        if actual_load.size == 0:
            return None

        sample_count = int(block_end - block_start + 1)

        sigma_idle = float(measurement.get("sigma_idle", 0.0) or 0.0)
        delta_mrr = float(measurement.get("delta_mrr", 0.0) or 0.0)
        steady_stats = self._evaluate_measurement_steady_gate(
            actual_load,
            sigma_idle=sigma_idle,
            sample_count=sample_count,
            min_sample_count=1,
        )
        actual_load_mean = float(steady_stats.get("p_meas", float("nan")))
        actual_load_var = float(steady_stats.get("actual_load_var", 0.0))
        actual_load_std = float(steady_stats.get("actual_load_std", 0.0))
        actual_load_diff_std = float(steady_stats.get("actual_load_diff_std", 0.0))
        variance_limit = float(steady_stats.get("variance_limit", float("inf")))
        diff_std_limit = float(steady_stats.get("diff_std_limit", float("inf")))
        steady_pass = bool(steady_stats.get("steady_pass", False))

        kc_values = np.asarray(measurement.get("kc_point", []), dtype=float)
        kc_valid_mask = np.asarray(measurement.get("kc_valid_mask", []), dtype=bool)
        kc_gated_out_mask = np.asarray(measurement.get("kc_gated_out_mask", []), dtype=bool)
        idle_power = np.asarray(measurement.get("predicted_idle_power", []), dtype=float)
        ap_values = np.asarray(measurement.get("mapped_ap", []), dtype=float)
        mrr_values = np.asarray(measurement.get("mapped_mrr", []), dtype=float)

        valid_kc_values = np.array([], dtype=float)
        valid_kc_count = 0
        gated_out_count = 0
        p_meas_value = float(actual_load_mean)
        p_idle_mean = float("nan")
        ap_mean = float("nan")
        mrr_mean = float("nan")
        if kc_values.size > block_end and kc_valid_mask.size > block_end:
            block_kc_values = kc_values[sample_slice]
            block_valid_mask = kc_valid_mask[sample_slice]
            valid_kc_values = block_kc_values[block_valid_mask & np.isfinite(block_kc_values)]
            valid_kc_count = int(valid_kc_values.size)
        if kc_gated_out_mask.size > block_end:
            gated_out_count = int(np.sum(kc_gated_out_mask[sample_slice]))

        kc_hat, sigma_kc, _ = self._summarize_interval_kc_statistics(valid_kc_values)
        if idle_power.size > block_end and ap_values.size > block_end and mrr_values.size > block_end:
            block_actual = np.asarray(actual_load_raw, dtype=float)
            block_idle = idle_power[sample_slice]
            block_ap = ap_values[sample_slice]
            block_mrr = mrr_values[sample_slice]
            reverse_mask = (
                kc_valid_mask[sample_slice]
                & np.isfinite(block_actual)
                & np.isfinite(block_idle)
                & np.isfinite(block_ap)
                & np.isfinite(block_mrr)
                & (block_mrr > 1e-12)
            )
            if np.any(reverse_mask):
                p_meas_value = float(np.mean(block_actual[reverse_mask]))
                p_idle_mean = float(np.mean(block_idle[reverse_mask]))
                ap_mean = float(np.mean(block_ap[reverse_mask]))
                mrr_mean = float(np.mean(block_mrr[reverse_mask]))
        if not np.isfinite(sigma_kc):
            sigma_kc = 0.0 if valid_kc_count else float("nan")

        return {
            "block_start": int(block_start),
            "block_end": int(block_end),
            "sample_count": int(sample_count),
            "p_meas": float(p_meas_value),
            "actual_load_var": float(actual_load_var),
            "actual_load_std": float(actual_load_std),
            "actual_load_diff_std": float(actual_load_diff_std),
            "variance_limit": variance_limit,
            "diff_std_limit": diff_std_limit,
            "steady_pass": bool(steady_pass),
            "sigma_idle": float(sigma_idle),
            "delta_mrr": float(delta_mrr),
            "valid_kc_count": int(valid_kc_count),
            "gated_out_count": int(gated_out_count),
            "kc_hat": float(kc_hat) if np.isfinite(kc_hat) else float("nan"),
            "sigma_kc": float(sigma_kc),
            "kc_source": "measurement_mode" if valid_kc_count > 0 and np.isfinite(kc_hat) else "",
            "p_idle_mean": float(p_idle_mean) if np.isfinite(p_idle_mean) else float("nan"),
            "a_p_mean": float(ap_mean) if np.isfinite(ap_mean) else float("nan"),
            "mrr_mean": float(mrr_mean) if np.isfinite(mrr_mean) else float("nan"),
        }

    def finalize_interval_kc(self, intervals):
        """为区间补齐 Kc / UCB；无有效样本时继承邻域或回退到全局参数。"""
        if not intervals:
            return intervals

        beta = float(self.kc_beta.get())
        global_kc = self.get_kc_value()
        global_sigma = float(self.kc_sigma.get())
        valid_indices = []
        for idx, interval in enumerate(intervals):
            if not self._record_represents_steady_interval(interval):
                for key in ("K_c_hat", "sigma_Kc", "K_c_UCB"):
                    interval.pop(key, None)
                interval["kc_source"] = ""
                continue
            kc_hat = interval.get("K_c_hat")
            sigma_kc = interval.get("sigma_Kc")
            if np.isfinite(kc_hat) and interval.get("valid_kc_count", 0) > 0:
                interval["K_c_hat"] = float(kc_hat)
                interval["sigma_Kc"] = float(max(float(sigma_kc or 0.0), 0.0))
                interval["K_c_UCB"] = float(interval["K_c_hat"] + beta * interval["sigma_Kc"])
                interval["kc_source"] = str(interval.get("kc_source") or "measurement_mode")
                valid_indices.append(idx)

        for idx, interval in enumerate(intervals):
            if not self._record_represents_steady_interval(interval):
                continue
            if str(interval.get("kc_source", "")).startswith("interval") or interval.get("kc_source") == "idle":
                continue
            if valid_indices:
                nearest_idx = min(valid_indices, key=lambda ref_idx: abs(ref_idx - idx))
                nearest = intervals[nearest_idx]
                fallback_kc = float(nearest.get("K_c_hat", global_kc))
                fallback_sigma = float(nearest.get("sigma_Kc", global_sigma))
                interval["kc_source"] = f"inherit:{nearest.get('zone_id', '')}"
            else:
                fallback_kc = float(global_kc)
                fallback_sigma = global_sigma
                interval["kc_source"] = "global"
            interval["K_c_hat"] = float(fallback_kc)
            interval["sigma_Kc"] = float(max(fallback_sigma, 0.0))
            interval["K_c_UCB"] = float(interval["K_c_hat"] + beta * interval["sigma_Kc"])
        return intervals

    def _reload_smif_pit_tree(self):
        tree = getattr(self, "smif_pit_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for entry in self._get_current_interval_records(allow_profile_fallback=False):
            values = []
            for key in tree["columns"]:
                value = entry.get(key)
                if isinstance(value, float):
                    values.append(f"{value:.6f}" if abs(value) < 1000 else f"{value:.3f}")
                else:
                    values.append("" if value is None else value)
            tree.insert("", "end", values=values)

    def _resolve_smif_display_target(self, raw_point_count):
        try:
            count = int(raw_point_count)
        except Exception:
            count = 0
        if count <= 0:
            return 0
        if count <= 12000:
            return count
        return min(12000, max(4000, int(2500 + 20.0 * math.sqrt(count))))

    def _resolve_smif_view_display_target(self, raw_point_count, visible_point_count=None):
        base_target = self._resolve_smif_display_target(raw_point_count)
        try:
            visible_count = int(visible_point_count) if visible_point_count is not None else int(raw_point_count)
        except Exception:
            visible_count = int(raw_point_count)
        if visible_count <= 0:
            return 0
        if visible_count <= 12000:
            return visible_count
        if raw_point_count <= 0 or visible_count >= raw_point_count:
            return min(visible_count, base_target)

        visible_ratio = max(float(visible_count) / float(raw_point_count), 1e-6)
        detail_boost = min(4.0, max(1.0, 1.0 / math.sqrt(visible_ratio)))
        boosted_target = int(round(base_target * detail_boost))
        return min(visible_count, max(base_target, boosted_target, 6000), 30000)

    def _compress_smif_path_indices(self, coords_segment, max_points):
        try:
            limit = int(max_points)
        except Exception:
            limit = 0
        coords = np.asarray(coords_segment, dtype=float)
        point_count = len(coords)
        if limit <= 0 or point_count <= limit or point_count <= 4:
            return np.arange(point_count, dtype=int)
        if limit == 1:
            return np.asarray([0], dtype=int)
        if limit == 2:
            return np.asarray([0, point_count - 1], dtype=int)

        bucket_count = max(1, min(point_count, limit // 8))
        if bucket_count <= 1:
            return np.asarray([0, point_count - 1], dtype=int)

        edges = np.linspace(0, point_count, num=bucket_count + 1, dtype=int)
        kept_indices = []
        last_idx = -1

        def _append(idx):
            nonlocal last_idx
            idx = int(idx)
            if idx < 0 or idx >= point_count or idx == last_idx:
                return
            kept_indices.append(idx)
            last_idx = idx

        for bucket_idx in range(bucket_count):
            start = int(edges[bucket_idx])
            end = int(edges[bucket_idx + 1])
            if end <= start:
                continue
            local_indices = {start, end - 1}
            local_coords = coords[start:end]
            for axis_idx in range(3):
                axis_vals = local_coords[:, axis_idx]
                finite_local = np.flatnonzero(np.isfinite(axis_vals))
                if finite_local.size == 0:
                    continue
                finite_values = axis_vals[finite_local]
                local_indices.add(int(start + finite_local[int(np.argmin(finite_values))]))
                local_indices.add(int(start + finite_local[int(np.argmax(finite_values))]))
            for idx in sorted(local_indices):
                _append(idx)

        _append(point_count - 1)
        if len(kept_indices) >= point_count:
            return np.arange(point_count, dtype=int)
        return np.asarray(kept_indices, dtype=int)

    def _collect_smif_path_blocks(self, row_indices, state_codes):
        row_arr = np.asarray(row_indices, dtype=int)
        state_arr = np.asarray(state_codes, dtype=np.int8)
        if row_arr.size == 0 or state_arr.size != row_arr.size:
            return []
        blocks = []
        block_start = 0
        block_state = int(state_arr[0])
        for idx in range(1, row_arr.size):
            is_contiguous = row_arr[idx] == row_arr[idx - 1] + 1
            if is_contiguous and int(state_arr[idx]) == block_state:
                continue
            blocks.append((int(block_start), int(idx - 1), int(block_state)))
            block_start = idx
            block_state = int(state_arr[idx])
        blocks.append((int(block_start), int(row_arr.size - 1), int(block_state)))
        return blocks

    def _collect_smif_display_points(self):
        cached_points = getattr(self, "_smif_interaction_points", None)
        if isinstance(cached_points, np.ndarray) and cached_points.ndim == 2 and cached_points.shape[1] == 3 and cached_points.size > 0:
            return [tuple(point) for point in cached_points]

        points = []

        if self.data:
            for row in self.data:
                try:
                    x_val = float(row.get("x"))
                    y_val = float(row.get("y"))
                    z_val = float(row.get("z"))
                except Exception:
                    continue
                if np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val):
                    points.append((x_val, y_val, z_val))

        profile = getattr(self, "gcode_profile", None)
        if not profile:
            return points

        segments = profile.get("trajectory_segments", [])
        usable_segments = self._get_smif_display_segments(segments)
        for segment in usable_segments:
            for prefix in ("start", "end"):
                try:
                    x_val = float(segment.get(f"{prefix}_x"))
                    y_val = float(segment.get(f"{prefix}_y"))
                    z_val = float(segment.get(f"{prefix}_z"))
                except Exception:
                    continue
                if np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val):
                    points.append((x_val, y_val, z_val))
        return points

    def _style_smif_axes(self, ax):
        ax.grid(False)
        ax.set_anchor('C')
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.pane.set_facecolor(SMIF_PANE_BG)
                axis.pane.set_edgecolor(SMIF_PANE_EDGE)
            except Exception:
                pass
            try:
                axis._axinfo["grid"]["linewidth"] = 0.0
                axis._axinfo["grid"]["color"] = (1.0, 1.0, 1.0, 0.0)
                axis._axinfo["axisline"]["color"] = SMIF_PANE_EDGE
                axis._axinfo["tick"]["color"] = SMIF_TEXT_MUTED
            except Exception:
                pass
        ax.tick_params(colors=SMIF_TEXT_COLOR, labelsize=max(PLOT_FONT_BASE - 1, 8), pad=1)

    def _style_smif_panel_axes(self, ax, title="", xlabel="", ylabel=""):
        if ax is None:
            return
        ax.set_facecolor(SMIF_PANEL_BG)
        for spine in ax.spines.values():
            try:
                spine.set_color(SMIF_PANEL_EDGE)
                spine.set_linewidth(0.9)
            except Exception:
                pass
        ax.tick_params(colors=SMIF_TEXT_COLOR, labelsize=max(PLOT_FONT_BASE - 2, 8))
        ax.grid(True, color=SMIF_GRID_COLOR, linewidth=0.6)
        ax.set_title(str(title), fontsize=PLOT_FONT_BASE, color=SMIF_TEXT_COLOR, pad=6)
        ax.set_xlabel(str(xlabel), color=SMIF_TEXT_COLOR, fontsize=max(PLOT_FONT_BASE - 1, 9), labelpad=4)
        ax.set_ylabel(str(ylabel), color=SMIF_TEXT_COLOR, fontsize=max(PLOT_FONT_BASE - 1, 9), labelpad=4)

    def _style_smif_main_axes(self, ax, title="", xlabel="", ylabel=""):
        if ax is None:
            return
        ax.set_facecolor(SMIF_AX_BG)
        for spine in ax.spines.values():
            try:
                spine.set_color(SMIF_PANE_EDGE)
                spine.set_linewidth(0.9)
            except Exception:
                pass
        ax.tick_params(colors=SMIF_TEXT_COLOR, labelsize=max(PLOT_FONT_BASE - 1, 8))
        ax.grid(True, color=SMIF_GRID_COLOR, linewidth=0.55, alpha=0.55)
        ax.set_title(str(title), fontsize=PLOT_FONT_BASE + 2, color=SMIF_TEXT_COLOR, pad=10, loc="left")
        ax.set_xlabel(str(xlabel), color=SMIF_TEXT_COLOR, fontsize=max(PLOT_FONT_BASE, 10), labelpad=6)
        ax.set_ylabel(str(ylabel), color=SMIF_TEXT_COLOR, fontsize=max(PLOT_FONT_BASE, 10), labelpad=6)
        try:
            ax.set_aspect("auto")
        except Exception:
            pass

    def _get_smif_axis_label(self, axis_idx):
        return {
            0: "X (mm)",
            1: "Y (mm)",
            2: "Z (mm)",
        }.get(int(axis_idx), f"Axis {axis_idx}")

    def _get_smif_projection_name(self, axis_pair):
        letters = "XYZ"
        try:
            return "".join(letters[int(idx)] for idx in axis_pair[:2])
        except Exception:
            return "Projection"

    def _get_smif_active_bounds(self):
        bounds = getattr(self, "_smif_focus_bounds", None) if self._get_smif_view_mode() == "focus" else None
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            bounds = getattr(self, "_smif_bounds", None)
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            return None
        try:
            mins = np.asarray(bounds[0], dtype=float)
            maxs = np.asarray(bounds[1], dtype=float)
        except Exception:
            return None
        if mins.shape != (3,) or maxs.shape != (3,):
            return None
        if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
            return None
        return mins, maxs

    def _has_smif_trajectory_source(self):
        profile = getattr(self, "gcode_profile", None)
        if not isinstance(profile, dict):
            return False
        segments = profile.get("trajectory_segments", [])
        return bool(segments)

    def _get_smif_display_segments(self, segments):
        usable_segments = [segment for segment in (segments or []) if isinstance(segment, dict)]
        if not usable_segments:
            return []

        # origin 推导出来的未就绪伪段只用于内部路径对齐，不应进入 SMIF 可视化。
        display_ready_segments = [segment for segment in usable_segments if bool(segment.get("display_ok", True))]
        if display_ready_segments:
            usable_segments = display_ready_segments

        non_initial_segments = [segment for segment in usable_segments if not bool(segment.get("is_initial_jump"))]
        if non_initial_segments:
            return non_initial_segments
        return usable_segments

    def _select_smif_projection_pairs(self):
        candidate_pairs = [(0, 1), (0, 2), (1, 2)]
        bounds = self._get_smif_active_bounds()
        if bounds is None:
            return (0, 2), [(0, 1), (1, 2)]

        mins, maxs = bounds
        spans = np.maximum(np.asarray(maxs - mins, dtype=float), 1e-9)
        scored_pairs = []
        for pair in candidate_pairs:
            area_score = float(spans[pair[0]] * spans[pair[1]])
            scored_pairs.append((area_score, pair))
        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        ordered_pairs = [pair for _score, pair in scored_pairs]
        main_pair = ordered_pairs[0] if ordered_pairs else (0, 2)
        side_pairs = ordered_pairs[1:] if len(ordered_pairs) > 1 else [(0, 1), (1, 2)]
        while len(side_pairs) < 2:
            for pair in candidate_pairs:
                if pair != main_pair and pair not in side_pairs:
                    side_pairs.append(pair)
                if len(side_pairs) >= 2:
                    break
        return main_pair, side_pairs[:2]

    def _normalize_smif_dashboard_block(self, block):
        if isinstance(block, dict):
            coords = np.asarray(block.get("coords"), dtype=float)
            metrics = np.asarray(block.get("metrics", []), dtype=float)
            rows = np.asarray(block.get("row_indices", []), dtype=int)
            try:
                state = int(block.get("state", -1))
            except Exception:
                state = -1
        else:
            coords = np.asarray(block, dtype=float)
            metrics = np.empty((0,), dtype=float)
            rows = np.empty((0,), dtype=int)
            state = -1
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.size == 0:
            return None
        finite_mask = np.all(np.isfinite(coords), axis=1)
        if not np.any(finite_mask):
            return None
        coords = coords[finite_mask]
        if metrics.size:
            metrics = metrics[:len(finite_mask)][finite_mask[:len(metrics)]]
        if rows.size:
            rows = rows[:len(finite_mask)][finite_mask[:len(rows)]]
        return coords, metrics, rows, state

    def _apply_smif_projection_limits(self, ax, axis_pair, bounds=None, emphasize=False):
        if ax is None:
            return
        if bounds is None:
            bounds = self._get_smif_active_bounds()
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            return
        try:
            mins = np.asarray(bounds[0], dtype=float)
            maxs = np.asarray(bounds[1], dtype=float)
        except Exception:
            return
        if mins.shape != (3,) or maxs.shape != (3,):
            return

        x_idx, y_idx = int(axis_pair[0]), int(axis_pair[1])
        x_span = max(float(maxs[x_idx] - mins[x_idx]), 1.0)
        y_span = max(float(maxs[y_idx] - mins[y_idx]), 1.0)
        x_margin_ratio = 0.03 if emphasize else 0.05
        y_margin_ratio = 0.05 if emphasize else 0.08
        x_margin = max(x_span * x_margin_ratio, 0.35 if emphasize else 0.60)
        y_margin = max(y_span * y_margin_ratio, 0.35 if emphasize else 0.60)
        ax.set_xlim(float(mins[x_idx] - x_margin), float(maxs[x_idx] + x_margin))
        ax.set_ylim(float(mins[y_idx] - y_margin), float(maxs[y_idx] + y_margin))

    def _draw_smif_projection_axis(self, ax, axis_pair, metric_key, has_colored_intervals, title="", emphasize=False):
        if ax is None:
            return

        axis_pair = (int(axis_pair[0]), int(axis_pair[1]))
        xlabel = self._get_smif_axis_label(axis_pair[0])
        ylabel = self._get_smif_axis_label(axis_pair[1])
        if emphasize:
            self._style_smif_main_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        else:
            self._style_smif_panel_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)

        dashboard_payload = getattr(self, "_smif_dashboard_payload", None) or {}
        base_payload = getattr(self, "_smif_base_projection_payload", None) or {}
        base_blocks = list(base_payload.get("blocks") or [])
        dashboard_blocks = list(dashboard_payload.get("blocks") or [])
        dashboard_point_coords = np.asarray(dashboard_payload.get("point_coords", np.empty((0, 3), dtype=float)), dtype=float)
        dashboard_point_states = np.asarray(dashboard_payload.get("point_states", np.empty((0,), dtype=np.int8)), dtype=np.int8)
        dashboard_point_metrics = np.asarray(dashboard_payload.get("point_metrics", np.empty((0,), dtype=float)), dtype=float)
        cmap_name = str(dashboard_payload.get("cmap_name") or "turbo")
        cmap = plt.get_cmap(cmap_name)
        metric_range = dashboard_payload.get("metric_range") or (0.0, 1.0)
        try:
            norm = matplotlib.colors.Normalize(vmin=float(metric_range[0]), vmax=float(metric_range[1]))
        except Exception:
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        has_nonsteady_point_overlay = bool(
            dashboard_point_coords.ndim == 2
            and dashboard_point_coords.shape[1] == 3
            and dashboard_point_coords.size > 0
            and np.any(dashboard_point_states[:len(dashboard_point_coords)] == 0)
        )

        line_segments = []
        line_colors = []
        line_widths = []
        scatter_coords = []
        scatter_colors = []
        scatter_sizes = []
        interval_line_width = float(self._get_smif_interval_line_width())
        interval_point_size = float(self._get_smif_interval_point_size())

        base_scale = 1.45 if emphasize else 1.0

        def _append_blocks(blocks, base_only=False):
            for block in blocks:
                normalized = self._normalize_smif_dashboard_block(block)
                if normalized is None:
                    continue
                coords, metrics, _rows, state = normalized
                proj = coords[:, [axis_pair[0], axis_pair[1]]]
                if len(proj) == 1:
                    scatter_coords.append(proj[0])
                    if base_only or state < 0:
                        scatter_colors.append(matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.30))
                        scatter_sizes.append(interval_point_size * base_scale)
                    elif state == 1:
                        scatter_colors.append(matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.94))
                        scatter_sizes.append(interval_point_size * base_scale)
                    elif state == 2:
                        metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                        scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                        scatter_sizes.append(interval_point_size * base_scale)
                    else:
                        if has_nonsteady_point_overlay and not base_only:
                            continue
                        metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                        scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                        scatter_sizes.append(max(interval_point_size * 0.55 * base_scale, 4.0))
                    continue

                if base_only or state < 0:
                    local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                    line_segments.extend(local_segments)
                    base_rgba = matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.26)
                    line_colors.extend([base_rgba] * len(local_segments))
                    line_widths.extend([1.0 * base_scale] * len(local_segments))
                elif state == 1:
                    local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                    line_segments.extend(local_segments)
                    idle_rgba = matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.96)
                    line_colors.extend([idle_rgba] * len(local_segments))
                    line_widths.extend([interval_line_width * base_scale] * len(local_segments))
                elif state == 2:
                    local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                    line_segments.extend(local_segments)
                    metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                    steady_color = cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR
                    line_colors.extend([steady_color] * len(local_segments))
                    line_widths.extend([interval_line_width * base_scale] * len(local_segments))
                else:
                    if has_nonsteady_point_overlay and not base_only:
                        continue
                    fallback_metrics = metrics if metrics.size else np.full(len(proj), np.nan, dtype=float)
                    for point_idx, point in enumerate(proj):
                        scatter_coords.append(point)
                        metric_value = float(fallback_metrics[min(point_idx, len(fallback_metrics) - 1)]) if len(fallback_metrics) else float("nan")
                        scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                        scatter_sizes.append(max(interval_point_size * 0.55 * base_scale, 4.0))

        _append_blocks(base_blocks, base_only=True)
        if has_colored_intervals:
            _append_blocks(dashboard_blocks, base_only=False)

        if has_nonsteady_point_overlay:
            nonsteady_mask = dashboard_point_states[:len(dashboard_point_coords)] == 0
            if np.any(nonsteady_mask):
                nonsteady_coords = dashboard_point_coords[nonsteady_mask][:, [axis_pair[0], axis_pair[1]]]
                nonsteady_metrics = dashboard_point_metrics[:len(nonsteady_mask)][nonsteady_mask[:len(dashboard_point_metrics)]]
                for point_idx, point in enumerate(nonsteady_coords):
                    scatter_coords.append(point)
                    metric_value = float(nonsteady_metrics[point_idx]) if point_idx < len(nonsteady_metrics) else float("nan")
                    scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                    scatter_sizes.append(interval_point_size * base_scale)

        if line_segments:
            ax.add_collection(
                LineCollection(
                    line_segments,
                    colors=line_colors,
                    linewidths=line_widths,
                    capstyle="butt",
                    joinstyle="miter",
                    zorder=3,
                )
            )
        if scatter_coords:
            scatter_arr = np.asarray(scatter_coords, dtype=float)
            ax.scatter(
                scatter_arr[:, 0],
                scatter_arr[:, 1],
                c=scatter_colors,
                s=scatter_sizes,
                marker="o",
                edgecolors="none",
                zorder=5,
            )

        self._apply_smif_projection_limits(ax, axis_pair, emphasize=emphasize)

    def _render_smif_main_panel(self, metric_key, has_colored_intervals):
        ax = getattr(self, "ax_smif", None)
        if ax is None:
            return
        main_pair, side_pairs = self._select_smif_projection_pairs()
        self._smif_main_axis_pair = main_pair
        self._smif_side_axis_pairs = side_pairs
        title = f"SMIF / {self._get_smif_projection_name(main_pair)} 主轨迹"
        self._draw_smif_projection_axis(
            ax,
            main_pair,
            metric_key,
            has_colored_intervals,
            title=title,
            emphasize=True,
        )

    def _build_smif_dashboard_axes(self):
        fig = getattr(self, "fig_smif", None)
        if fig is None:
            return
        fig.clf()
        fig.patch.set_facecolor(SMIF_FIG_BG)
        try:
            gs = fig.add_gridspec(
                1,
                1,
                left=0.015,
                right=0.955,
                top=0.985,
                bottom=0.035,
            )
        except Exception:
            gs = fig.add_gridspec(1, 1)
        self.ax_smif = fig.add_subplot(gs[0, 0], projection="3d")
        self.ax_smif_xy = None
        self.ax_smif_xz = None
        self.ax_smif_metric = None
        self._smif_colorbar_ax = fig.add_axes([0.962, 0.18, 0.012, 0.56])
        self._smif_colorbar_ax.set_visible(False)
        self._smif_colorbar_ax.set_facecolor(SMIF_FIG_BG)

        self.ax_smif.set_facecolor(SMIF_AX_BG)
        self.ax_smif.set_title("SMIF / 3D G代码轨迹", fontsize=PLOT_FONT_BASE + 3, color=SMIF_TEXT_COLOR, pad=10)
        self.ax_smif.set_xlabel("X (mm)", color=SMIF_TEXT_COLOR, labelpad=8)
        self.ax_smif.set_ylabel("Y (mm)", color=SMIF_TEXT_COLOR, labelpad=8)
        self.ax_smif.set_zlabel("Z (mm)", color=SMIF_TEXT_COLOR, labelpad=6)
        self._style_smif_axes(self.ax_smif)
        try:
            self.ax_smif.set_proj_type("ortho")
        except Exception:
            pass
        self.ax_smif.view_init(elev=SMIF_MAIN_ELEV, azim=SMIF_MAIN_AZIM)
        self._set_smif_viewer_box(self.ax_smif)
        self.ax_smif.set_anchor("C")

    def _build_smif_summary_lines(self, metric_key, has_colored_intervals):
        lines = []
        view_mode = "区间聚焦" if self._get_smif_view_mode() == "focus" else "完整轨迹"
        lines.append(f"视图: {view_mode}")

        bounds = getattr(self, "_smif_focus_bounds", None) if self._get_smif_view_mode() == "focus" else None
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            bounds = getattr(self, "_smif_bounds", None)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            try:
                spans = np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float)
                if spans.shape == (3,) and np.all(np.isfinite(spans)):
                    lines.append(f"跨度: XΔ{spans[0]:.2f}  YΔ{spans[1]:.2f}  ZΔ{spans[2]:.2f}")
            except Exception:
                pass

        if has_colored_intervals:
            payload = getattr(self, "_smif_dashboard_payload", None)
            lines.append(f"区间: {len((payload or {}).get('interval_records') or [])} 段")
        else:
            base_payload = getattr(self, "_smif_base_projection_payload", None) or {}
            base_blocks = base_payload.get("blocks") or []
            if base_blocks:
                lines.append(f"轨迹段: {len(base_blocks)}")
            lines.append("状态: 等待区间 Kc")
        return lines

    def _draw_smif_summary_box(self, metric_key, has_colored_intervals):
        ax = getattr(self, "ax_smif", None)
        if ax is None:
            return
        summary_lines = self._build_smif_summary_lines(metric_key, has_colored_intervals)
        note_text_color = "#111111"
        note_facecolor = (1.0, 1.0, 1.0, 0.96)
        note_edgecolor = (0.20, 0.20, 0.20, 0.55)
        if summary_lines:
            ax.text2D(
                0.02,
                0.98,
                "\n".join(summary_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=max(PLOT_FONT_BASE + 1, 11),
                color=note_text_color,
                bbox=dict(
                    boxstyle="round,pad=0.48",
                    facecolor=note_facecolor,
                    edgecolor=note_edgecolor,
                    linewidth=1.1,
                ),
            )

        legend_line_width = max(float(self._get_smif_interval_line_width()) * 1.35, 1.0)
        legend_marker_size = max(float(self._get_smif_interval_point_size()) * 0.44, 4.0)
        summary_line_count = max(len(summary_lines), 1)
        legend_anchor_y = max(0.16, 0.98 - 0.078 * float(summary_line_count))
        if has_colored_intervals:
            legend_handles = [
                Line2D([0], [0], color=SMIF_IDLE_COLOR, lw=legend_line_width, label="空载/基线"),
                Line2D([0], [0], linestyle="None", marker="o", markersize=legend_marker_size, markerfacecolor=SMIF_NONSTEADY_COLOR, markeredgewidth=0.0, label="非稳态点"),
                Line2D([0], [0], color="#FFD166", lw=legend_line_width, label="稳态细线 / Kc"),
            ]
        else:
            legend_handles = [Line2D([0], [0], color=SMIF_IDLE_COLOR, lw=legend_line_width, label="NC轨迹")]
        try:
            legend = ax.legend(
                handles=legend_handles,
                loc="upper left",
                bbox_to_anchor=(0.02, legend_anchor_y),
                frameon=True,
                fontsize=max(PLOT_FONT_BASE, 10),
                borderpad=0.65,
                labelspacing=0.45,
                handlelength=2.2,
            )
            frame = legend.get_frame()
            frame.set_facecolor(note_facecolor)
            frame.set_edgecolor(note_edgecolor)
            frame.set_linewidth(1.0)
            for text in legend.get_texts():
                text.set_color(note_text_color)
        except Exception:
            pass

    def _render_smif_side_panels(self, metric_key, has_colored_intervals):
        xy_ax = getattr(self, "ax_smif_xy", None)
        xz_ax = getattr(self, "ax_smif_xz", None)
        metric_ax = getattr(self, "ax_smif_metric", None)
        if xy_ax is None or xz_ax is None or metric_ax is None:
            return

        self._style_smif_panel_axes(xy_ax, title="XY 投影", xlabel="X (mm)", ylabel="Y (mm)")
        self._style_smif_panel_axes(xz_ax, title="XZ 投影", xlabel="X (mm)", ylabel="Z (mm)")
        self._style_smif_panel_axes(metric_ax, title="Kc 剖面 / 状态", xlabel="样本索引", ylabel="Kc")

        dashboard_payload = getattr(self, "_smif_dashboard_payload", None) or {}
        base_payload = getattr(self, "_smif_base_projection_payload", None) or {}
        base_blocks = list(base_payload.get("blocks") or [])
        dashboard_blocks = list(dashboard_payload.get("blocks") or [])
        dashboard_point_coords = np.asarray(dashboard_payload.get("point_coords", np.empty((0, 3), dtype=float)), dtype=float)
        dashboard_point_states = np.asarray(dashboard_payload.get("point_states", np.empty((0,), dtype=np.int8)), dtype=np.int8)
        dashboard_point_metrics = np.asarray(dashboard_payload.get("point_metrics", np.empty((0,), dtype=float)), dtype=float)
        cmap_name = str(dashboard_payload.get("cmap_name") or "turbo")
        cmap = plt.get_cmap(cmap_name)
        metric_range = dashboard_payload.get("metric_range") or (0.0, 1.0)
        try:
            norm = matplotlib.colors.Normalize(vmin=float(metric_range[0]), vmax=float(metric_range[1]))
        except Exception:
            norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
        has_nonsteady_point_overlay = bool(
            dashboard_point_coords.ndim == 2
            and dashboard_point_coords.shape[1] == 3
            and dashboard_point_coords.size > 0
            and np.any(dashboard_point_states[:len(dashboard_point_coords)] == 0)
        )

        def _normalize_block(block):
            if isinstance(block, dict):
                coords = np.asarray(block.get("coords"), dtype=float)
                metrics = np.asarray(block.get("metrics", []), dtype=float)
                rows = np.asarray(block.get("row_indices", []), dtype=int)
                try:
                    state = int(block.get("state", -1))
                except Exception:
                    state = -1
            else:
                coords = np.asarray(block, dtype=float)
                metrics = np.empty((0,), dtype=float)
                rows = np.empty((0,), dtype=int)
                state = -1
            if coords.ndim != 2 or coords.shape[1] != 3 or coords.size == 0:
                return None
            finite_mask = np.all(np.isfinite(coords), axis=1)
            if not np.any(finite_mask):
                return None
            coords = coords[finite_mask]
            if metrics.size:
                metrics = metrics[:len(finite_mask)][finite_mask[:len(metrics)]]
            if rows.size:
                rows = rows[:len(finite_mask)][finite_mask[:len(rows)]]
            return coords, metrics, rows, state

        def _draw_projection(ax, axis_pair):
            line_segments = []
            line_colors = []
            line_widths = []
            scatter_coords = []
            scatter_colors = []
            scatter_sizes = []

            def _append_blocks(blocks, base_only=False):
                for block in blocks:
                    normalized = _normalize_block(block)
                    if normalized is None:
                        continue
                    coords, metrics, _rows, state = normalized
                    proj = coords[:, [axis_pair[0], axis_pair[1]]]
                    if len(proj) == 1:
                        scatter_coords.append(proj[0])
                        if base_only or state < 0:
                            scatter_colors.append(matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.28))
                            scatter_sizes.append(8)
                        elif state == 1:
                            scatter_colors.append(matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.92))
                            scatter_sizes.append(12)
                        elif state == 2:
                            metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                            scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                            scatter_sizes.append(9)
                        else:
                            if has_nonsteady_point_overlay and not base_only:
                                continue
                            metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                            scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                            scatter_sizes.append(max(interval_point_size * 0.55, 4.0))
                        continue

                    if base_only or state < 0:
                        local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                        line_segments.extend(local_segments)
                        base_rgba = matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.24)
                        line_colors.extend([base_rgba] * len(local_segments))
                        line_widths.extend([0.9] * len(local_segments))
                    elif state == 1:
                        local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                        line_segments.extend(local_segments)
                        idle_rgba = matplotlib.colors.to_rgba(SMIF_IDLE_COLOR, 0.92)
                        line_colors.extend([idle_rgba] * len(local_segments))
                        line_widths.extend([0.88] * len(local_segments))
                    elif state == 2:
                        local_segments = [[tuple(proj[idx]), tuple(proj[idx + 1])] for idx in range(len(proj) - 1)]
                        line_segments.extend(local_segments)
                        metric_value = float(np.nanmedian(metrics)) if metrics.size else float("nan")
                        steady_color = cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR
                        line_colors.extend([steady_color] * len(local_segments))
                        line_widths.extend([0.92] * len(local_segments))
                    else:
                        if has_nonsteady_point_overlay and not base_only:
                            continue
                        fallback_metrics = metrics if metrics.size else np.full(len(proj), np.nan, dtype=float)
                        for point_idx, point in enumerate(proj):
                            scatter_coords.append(point)
                            metric_value = float(fallback_metrics[min(point_idx, len(fallback_metrics) - 1)]) if len(fallback_metrics) else float("nan")
                            scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                            scatter_sizes.append(max(interval_point_size * 0.55, 4.0))

            _append_blocks(base_blocks, base_only=True)
            if has_colored_intervals:
                _append_blocks(dashboard_blocks, base_only=False)

            if has_nonsteady_point_overlay:
                nonsteady_mask = dashboard_point_states[:len(dashboard_point_coords)] == 0
                if np.any(nonsteady_mask):
                    nonsteady_coords = dashboard_point_coords[nonsteady_mask][:, [axis_pair[0], axis_pair[1]]]
                    nonsteady_metrics = dashboard_point_metrics[:len(nonsteady_mask)][nonsteady_mask[:len(dashboard_point_metrics)]]
                    for point_idx, point in enumerate(nonsteady_coords):
                        scatter_coords.append(point)
                        metric_value = float(nonsteady_metrics[point_idx]) if point_idx < len(nonsteady_metrics) else float("nan")
                        scatter_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                        scatter_sizes.append(11)

            if line_segments:
                ax.add_collection(
                    LineCollection(
                        line_segments,
                        colors=line_colors,
                        linewidths=line_widths,
                        capstyle="butt",
                        joinstyle="miter",
                    )
                )
            if scatter_coords:
                scatter_arr = np.asarray(scatter_coords, dtype=float)
                ax.scatter(
                    scatter_arr[:, 0],
                    scatter_arr[:, 1],
                    c=scatter_colors,
                    s=scatter_sizes,
                    marker="o",
                    edgecolors="none",
                    zorder=5,
                )

        _draw_projection(xy_ax, (0, 1))
        _draw_projection(xz_ax, (0, 2))

        bounds = getattr(self, "_smif_focus_bounds", None) if self._get_smif_view_mode() == "focus" else None
        if not (isinstance(bounds, tuple) and len(bounds) == 2):
            bounds = getattr(self, "_smif_bounds", None)
        if isinstance(bounds, tuple) and len(bounds) == 2:
            try:
                mins = np.asarray(bounds[0], dtype=float)
                maxs = np.asarray(bounds[1], dtype=float)
                if mins.shape == (3,) and maxs.shape == (3,):
                    for ax, x_idx, y_idx in ((xy_ax, 0, 1), (xz_ax, 0, 2)):
                        x_span = max(float(maxs[x_idx] - mins[x_idx]), 1.0)
                        y_span = max(float(maxs[y_idx] - mins[y_idx]), 1.0)
                        x_margin = max(x_span * 0.05, 0.6)
                        y_margin = max(y_span * 0.08, 0.6)
                        ax.set_xlim(float(mins[x_idx] - x_margin), float(maxs[x_idx] + x_margin))
                        ax.set_ylim(float(mins[y_idx] - y_margin), float(maxs[y_idx] + y_margin))
            except Exception:
                pass

        if has_colored_intervals and dashboard_blocks:
            all_rows = []
            all_metrics = []
            all_states = []
            for block in dashboard_blocks:
                normalized = _normalize_block(block)
                if normalized is None:
                    continue
                _coords, metrics, rows, state = normalized
                if rows.size == 0 or metrics.size == 0:
                    continue
                all_rows.append(rows)
                all_metrics.append(metrics[:len(rows)])
                all_states.append(np.full(len(rows), state, dtype=int))
                try:
                    left = float(rows[0])
                    right = float(rows[-1])
                    if right <= left:
                        right = left + 1.0
                    fill_color = {
                        0: SMIF_METRIC_NONSTEADY_FILL,
                        1: SMIF_METRIC_IDLE_FILL,
                        2: SMIF_METRIC_STEADY_FILL,
                    }.get(int(state), SMIF_METRIC_IDLE_FILL)
                    metric_ax.axvspan(left, right, color=fill_color, alpha=0.11, linewidth=0.0, zorder=0)
                except Exception:
                    pass

            if all_rows and all_metrics and all_states:
                row_arr = np.concatenate(all_rows)
                metric_arr = np.concatenate(all_metrics)
                state_arr = np.concatenate(all_states)
                order = np.argsort(row_arr)
                row_arr = row_arr[order]
                metric_arr = metric_arr[order]
                state_arr = state_arr[order]
                if len(row_arr) > 2600:
                    keep_idx = np.unique(np.linspace(0, len(row_arr) - 1, num=2600, dtype=int))
                    row_arr = row_arr[keep_idx]
                    metric_arr = metric_arr[keep_idx]
                    state_arr = state_arr[keep_idx]

                metric_ax.plot(row_arr, metric_arr, color=(1.0, 1.0, 1.0, 0.24), linewidth=1.0, zorder=1)
                idle_mask = state_arr == 1
                if np.any(idle_mask):
                    metric_ax.scatter(
                        row_arr[idle_mask],
                        metric_arr[idle_mask],
                        c=SMIF_IDLE_COLOR,
                        s=10,
                        edgecolors="none",
                        alpha=0.88,
                        zorder=3,
                    )
                active_mask = ~idle_mask
                if np.any(active_mask):
                    metric_ax.scatter(
                        row_arr[active_mask],
                        metric_arr[active_mask],
                        c=metric_arr[active_mask],
                        cmap=cmap,
                        norm=norm,
                        s=14,
                        edgecolors="none",
                        alpha=0.95,
                        zorder=4,
                    )
                metric_ax.set_xlim(float(np.min(row_arr)), float(np.max(row_arr)) if len(row_arr) > 1 else float(row_arr[0] + 1.0))
                metric_min = float(np.min(metric_arr))
                metric_max = float(np.max(metric_arr))
                if not np.isfinite(metric_min) or not np.isfinite(metric_max):
                    metric_min, metric_max = 0.0, 1.0
                if metric_max <= metric_min:
                    metric_max = metric_min + 1.0
                metric_pad = max((metric_max - metric_min) * 0.08, 0.1)
                metric_ax.set_ylim(metric_min - metric_pad, metric_max + metric_pad)
                metric_ax.text(
                    0.02,
                    0.96,
                    f"{self._get_smif_metric_label(metric_key)} 剖面",
                    transform=metric_ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=max(PLOT_FONT_BASE - 2, 9),
                    color=SMIF_TEXT_COLOR,
                )
            else:
                metric_ax.text(
                    0.5,
                    0.5,
                    "当前区间缺少有效 Kc 数据",
                    transform=metric_ax.transAxes,
                    ha="center",
                    va="center",
                    color=SMIF_TEXT_MUTED,
                    fontsize=max(PLOT_FONT_BASE - 1, 9),
                )
                metric_ax.set_xticks([])
                metric_ax.set_yticks([])
        else:
            metric_ax.text(
                0.5,
                0.5,
                "等待区间 Kc 更新",
                transform=metric_ax.transAxes,
                ha="center",
                va="center",
                color=SMIF_TEXT_MUTED,
                fontsize=max(PLOT_FONT_BASE - 1, 9),
            )
            metric_ax.set_xticks([])
            metric_ax.set_yticks([])

    def _apply_smif_axis_limits(self, ax):
        if ax is None or not self._axis_is_3d(ax):
            return
        transform = self._resolve_smif_display_transform()
        if not isinstance(transform, dict):
            return
        display_ratios = np.asarray(transform.get("display_ratios", [1.0, 1.0, 1.0]), dtype=float)
        margins = np.maximum(display_ratios * 0.10, 0.16)
        self._smif_display_bounds = (
            -display_ratios - margins,
            display_ratios + margins,
        )
        ax.set_xlim3d(float(-display_ratios[0] - margins[0]), float(display_ratios[0] + margins[0]))
        ax.set_ylim3d(float(-display_ratios[1] - margins[1]), float(display_ratios[1] + margins[1]))
        ax.set_zlim3d(float(-display_ratios[2] - margins[2]), float(display_ratios[2] + margins[2]))
        self._refresh_smif_main_axis_ticks(ax)
        self._set_smif_viewer_box(ax)
        ax.set_anchor("C")

    def _draw_smif_empty_placeholder(self):
        self._build_smif_dashboard_axes()
        axes = [getattr(self, "ax_smif_xy", None), getattr(self, "ax_smif_xz", None), getattr(self, "ax_smif_metric", None)]
        for ax in axes:
            if ax is None:
                continue
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                "等待数据",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=PLOT_FONT_BASE,
                color=SMIF_TEXT_MUTED,
            )
        if getattr(self, "_smif_colorbar_ax", None) is not None:
            try:
                self._smif_colorbar_ax.set_visible(False)
                self._smif_colorbar_ax.set_xticks([])
                self._smif_colorbar_ax.set_yticks([])
            except Exception:
                pass
        ax3d = getattr(self, "ax_smif", None)
        if ax3d is not None:
            ax3d.text2D(
                0.5,
                0.52,
                "导入 G 代码 NC 后显示 3D 轨迹",
                ha="center",
                va="center",
                transform=ax3d.transAxes,
                fontsize=PLOT_FONT_BASE + 2,
                color=SMIF_TEXT_COLOR,
            )
            ax3d.text2D(
                0.5,
                0.45,
                "完成稳态区间与 Kc 更新后，自动生成多视图 SMIF 仪表板",
                ha="center",
                va="center",
                transform=ax3d.transAxes,
                fontsize=PLOT_FONT_BASE - 1,
                color=SMIF_TEXT_MUTED,
            )
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])

    def _plot_nc_trajectory(self, ax, max_segments=None, view_limits=None):
        profile = getattr(self, "gcode_profile", None)
        if not profile:
            self._smif_base_projection_payload = None
            return False
        segments = profile.get("trajectory_segments", [])
        if not segments:
            self._smif_base_projection_payload = None
            return False

        display_segments = self._get_smif_display_segments(segments)
        if not display_segments:
            display_segments = segments
        focus_segments, focus_bounds = self._resolve_smif_trajectory_focus_segments(display_segments)
        if focus_bounds is not None:
            self._smif_focus_bounds = focus_bounds
        if self._get_smif_view_mode() == "focus" and focus_segments:
            display_segments = focus_segments

        raw_segment_count = len(display_segments)
        can_draw_3d = self._axis_is_3d(ax)
        if raw_segment_count > 0:
            try:
                segment_limit = int(max_segments) if max_segments is not None else raw_segment_count
            except Exception:
                segment_limit = raw_segment_count
            if segment_limit > 0 and raw_segment_count > segment_limit:
                keep_indices = np.unique(np.linspace(0, raw_segment_count - 1, num=segment_limit, dtype=int))
                display_segments = [display_segments[int(idx)] for idx in keep_indices]

        xs = []
        ys = []
        zs = []
        base_projection_blocks = []
        interaction_points = []
        for segment in display_segments:
            try:
                block_coords = np.asarray(
                    [
                        [float(segment["start_x"]), float(segment["start_y"]), float(segment["start_z"])],
                        [float(segment["end_x"]), float(segment["end_y"]), float(segment["end_z"])],
                    ],
                    dtype=float,
                )
                if block_coords.shape == (2, 3) and np.all(np.isfinite(block_coords)):
                    base_projection_blocks.append(block_coords)
                    display_block = self._transform_smif_coords(block_coords)
                    if display_block.shape == (2, 3):
                        interaction_points.append(display_block)
                        if can_draw_3d:
                            xs.extend([display_block[0, 0], display_block[1, 0], np.nan])
                            ys.extend([display_block[0, 1], display_block[1, 1], np.nan])
                            zs.extend([display_block[0, 2], display_block[1, 2], np.nan])
            except Exception:
                pass

        if base_projection_blocks:
            finite_coords = np.asarray(
                [point for block in base_projection_blocks for point in block if np.all(np.isfinite(point))],
                dtype=float,
            )
            if finite_coords.size > 0:
                self._update_smif_bounds(finite_coords)
                if focus_bounds is None:
                    self._smif_focus_bounds = self._compute_smif_effective_bounds(finite_coords)
        else:
            self._smif_base_projection_payload = None
            return False

        visible_blocks = list(base_projection_blocks)
        if isinstance(view_limits, dict):
            try:
                xlim = tuple(sorted(float(v) for v in view_limits.get("xlim", ())))
                ylim = tuple(sorted(float(v) for v in view_limits.get("ylim", ())))
                zlim = tuple(sorted(float(v) for v in view_limits.get("zlim", ())))
            except Exception:
                xlim = ylim = zlim = ()
            if len(xlim) == len(ylim) == len(zlim) == 2:
                x_margin = max(abs(xlim[1] - xlim[0]) * 0.03, 0.05)
                y_margin = max(abs(ylim[1] - ylim[0]) * 0.03, 0.05)
                z_margin = max(abs(zlim[1] - zlim[0]) * 0.03, 0.05)
                filtered_blocks = []
                for block_coords in base_projection_blocks:
                    display_block = self._transform_smif_coords(block_coords)
                    if display_block.shape != (2, 3) or not np.all(np.isfinite(display_block)):
                        continue
                    seg_x_min = float(np.min(display_block[:, 0]))
                    seg_x_max = float(np.max(display_block[:, 0]))
                    seg_y_min = float(np.min(display_block[:, 1]))
                    seg_y_max = float(np.max(display_block[:, 1]))
                    seg_z_min = float(np.min(display_block[:, 2]))
                    seg_z_max = float(np.max(display_block[:, 2]))
                    intersects = not (
                        seg_x_max < xlim[0] - x_margin or seg_x_min > xlim[1] + x_margin
                        or seg_y_max < ylim[0] - y_margin or seg_y_min > ylim[1] + y_margin
                        or seg_z_max < zlim[0] - z_margin or seg_z_min > zlim[1] + z_margin
                    )
                    if intersects:
                        filtered_blocks.append(block_coords)
                if filtered_blocks:
                    visible_blocks = filtered_blocks

        xs = []
        ys = []
        zs = []
        interaction_points = []
        for block_coords in visible_blocks:
            display_block = self._transform_smif_coords(block_coords)
            if display_block.shape != (2, 3) or not np.all(np.isfinite(display_block)):
                continue
            interaction_points.append(display_block)
            if can_draw_3d:
                xs.extend([display_block[0, 0], display_block[1, 0], np.nan])
                ys.extend([display_block[0, 1], display_block[1, 1], np.nan])
                zs.extend([display_block[0, 2], display_block[1, 2], np.nan])

        if can_draw_3d and xs and ys and zs:
            ax.plot(xs, ys, zs, color=SMIF_IDLE_COLOR, linewidth=1.15, zorder=1, label="NC轨迹")
        if interaction_points:
            merged_display = np.vstack(interaction_points)
            anchor_target = min(len(merged_display), 3000)
            anchor_indices = self._compress_smif_path_indices(merged_display, anchor_target)
            self._smif_interaction_points = merged_display[anchor_indices]
        self._smif_base_projection_payload = {
            "blocks": base_projection_blocks,
            "focus_bounds": focus_bounds,
            "raw_segment_count": int(raw_segment_count),
        }
        return True

    def _resolve_smif_interval_path_span(self, record):
        if not isinstance(record, dict) or not self.data:
            return None

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        start_idx = None
        end_idx = None
        try:
            start_idx = int(record.get("start_idx"))
            end_idx = int(record.get("end_idx"))
        except Exception:
            bounds = self._resolve_interval_process_bounds(record)
            if bounds:
                try:
                    start_idx = int(bounds.get("start_idx"))
                    end_idx = int(bounds.get("end_idx"))
                except Exception:
                    start_idx = None
                    end_idx = None

        if start_idx is None or end_idx is None:
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        if start_idx < 0 or end_idx >= len(self.data):
            return None

        start_row = self.data[int(start_idx)]
        end_row = self.data[int(end_idx)]
        start_val = None
        end_val = None
        for key in ("path_start", "path_position", "path_end"):
            start_val = _safe_float(start_row.get(key))
            if start_val is not None:
                break
        for key in ("path_end", "path_position", "path_start"):
            end_val = _safe_float(end_row.get(key))
            if end_val is not None:
                break
        if start_val is None or end_val is None:
            return None
        return (min(start_val, end_val), max(start_val, end_val))

    def _resolve_smif_interval_line_span(self, record):
        if not isinstance(record, dict):
            return None
        start_line = record.get("start_line")
        end_line = record.get("end_line")
        if start_line is None or end_line is None:
            start_from_label, _ = self._parse_line_point_label(record.get("process_start_label"))
            end_from_label, _ = self._parse_line_point_label(record.get("process_end_label"))
            start_line = start_line if start_line is not None else start_from_label
            end_line = end_line if end_line is not None else end_from_label
        try:
            start_val = int(start_line)
            end_val = int(end_line)
        except Exception:
            return None
        return (min(start_val, end_val), max(start_val, end_val))

    def _is_smif_process_row_cutting(self, row):
        if not isinstance(row, dict):
            return False

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        move_type = str(row.get("type") or "").strip().lower()
        if move_type == "rapid":
            return False

        mrr_val = _safe_float(row.get("MRR"))
        if mrr_val is not None and mrr_val > 1e-9:
            return True

        ap_val = _safe_float(row.get("ap"))
        ae_val = _safe_float(row.get("ae"))
        feed_val = _safe_float(row.get("feed_effective"))
        if (
            ap_val is not None and ae_val is not None
            and ap_val > 1e-9 and ae_val > 1e-9
            and (feed_val is None or feed_val > 1e-9 or move_type == "cutting")
        ):
            return True
        return False

    def _resolve_smif_segment_default_state(self, segment):
        if not isinstance(segment, dict):
            return 1

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        motion_type = str(segment.get("motion_type") or "").strip().lower()
        if motion_type == "rapid":
            return 1

        spindle_on = bool(segment.get("spindle_on"))
        active_speed = _safe_float(segment.get("active_speed"))
        feed_value = _safe_float(segment.get("feed"))

        if not spindle_on:
            return 1
        if active_speed is not None and active_speed <= 1e-9:
            return 1
        if feed_value is not None and feed_value <= 1e-9 and motion_type != "cutting":
            return 1
        return 0

    def _resolve_smif_row_metric_value(self, row, row_index, metric_key, point_kc_map, fallback_metric, beta=0.0, global_sigma=0.0):
        if not isinstance(row, dict):
            return float("nan")

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        try:
            line_no = int(row.get("line_no_aligned", row_index))
        except Exception:
            line_no = int(row_index)
        try:
            point_idx = int(row.get("process_point_index", 0))
        except Exception:
            point_idx = 0

        kc_value = None
        if isinstance(point_kc_map, dict):
            mapped_value = point_kc_map.get((int(line_no), int(point_idx)))
            kc_value = _safe_float(mapped_value)
        if kc_value is None:
            kc_value = _safe_float(row.get("K_c", row.get("K", fallback_metric)))
        if kc_value is None:
            return float("nan")
        kc_value = max(float(kc_value), 0.0)
        if str(metric_key) == "K_c_UCB":
            return float(kc_value + max(float(beta), 0.0) * max(float(global_sigma), 0.0))
        return float(kc_value)

    def _resolve_smif_row_anchor_path(self, row):
        if not isinstance(row, dict):
            return None

        def _safe_float(value):
            try:
                numeric = float(value)
            except Exception:
                return None
            if not np.isfinite(numeric):
                return None
            return float(numeric)

        path_start = _safe_float(row.get("path_start"))
        path_end = _safe_float(row.get("path_end"))
        if path_start is not None and path_end is not None:
            if abs(path_end - path_start) <= 1e-9:
                return float(path_start)
            try:
                point_idx = max(int(row.get("process_point_index", 0)), 0)
            except Exception:
                point_idx = 0
            try:
                point_count = max(int(row.get("process_point_count", 1)), 1)
            except Exception:
                point_count = 1
            fraction = (float(point_idx) + 0.5) / max(float(point_count), 1.0)
            fraction = min(max(fraction, 0.0), 1.0)
            return float(path_start + (path_end - path_start) * fraction)

        for key in ("path_position", "path_end", "path_start"):
            value = _safe_float(row.get(key))
            if value is not None:
                return float(value)
        return None

    def _map_smif_process_points_to_segments(self, segments, rows, row_states, row_metrics):
        if not segments or not rows:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int8), np.empty((0,), dtype=float), np.empty((0,), dtype=int)

        segment_infos = []
        for segment in segments:
            try:
                start_point = np.asarray(
                    [float(segment.get("start_x")), float(segment.get("start_y")), float(segment.get("start_z"))],
                    dtype=float,
                )
                end_point = np.asarray(
                    [float(segment.get("end_x")), float(segment.get("end_y")), float(segment.get("end_z"))],
                    dtype=float,
                )
                path_start = float(segment.get("path_start"))
                path_end = float(segment.get("path_end"))
            except Exception:
                continue
            if not (
                start_point.shape == (3,)
                and end_point.shape == (3,)
                and np.all(np.isfinite(start_point))
                and np.all(np.isfinite(end_point))
                and np.isfinite(path_start)
                and np.isfinite(path_end)
            ):
                continue
            seg_min = float(min(path_start, path_end))
            seg_max = float(max(path_start, path_end))
            segment_infos.append({
                "segment": segment,
                "start_point": start_point,
                "end_point": end_point,
                "path_start": float(path_start),
                "path_end": float(path_end),
                "path_min": seg_min,
                "path_max": seg_max,
                "span": max(seg_max - seg_min, 1e-9),
            })

        if not segment_infos:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int8), np.empty((0,), dtype=float), np.empty((0,), dtype=int)

        point_coords = []
        point_states = []
        point_metrics = []
        point_rows = []
        for row_index, row in enumerate(rows):
            anchor_path = self._resolve_smif_row_anchor_path(row)
            if anchor_path is None or not np.isfinite(anchor_path):
                continue

            best_info = None
            best_distance = float("inf")
            best_span = float("inf")
            for info in segment_infos:
                if info["path_min"] - 1e-9 <= float(anchor_path) <= info["path_max"] + 1e-9:
                    distance = 0.0
                else:
                    distance = min(abs(float(anchor_path) - info["path_min"]), abs(float(anchor_path) - info["path_max"]))
                if distance < best_distance - 1e-9 or (
                    abs(distance - best_distance) <= 1e-9 and info["span"] < best_span
                ):
                    best_info = info
                    best_distance = float(distance)
                    best_span = float(info["span"])
            if best_info is None:
                continue

            path_delta = float(best_info["path_end"] - best_info["path_start"])
            if abs(path_delta) <= 1e-9:
                interp_t = 0.5
            else:
                interp_t = (float(anchor_path) - float(best_info["path_start"])) / path_delta
            interp_t = min(max(float(interp_t), 0.0), 1.0)
            coord = best_info["start_point"] + (best_info["end_point"] - best_info["start_point"]) * interp_t
            if not np.all(np.isfinite(coord)):
                continue

            point_coords.append(np.asarray(coord, dtype=float))
            point_states.append(int(row_states[row_index]))
            point_metrics.append(float(row_metrics[row_index]))
            point_rows.append(int(row_index))

        if not point_coords:
            return np.empty((0, 3), dtype=float), np.empty((0,), dtype=np.int8), np.empty((0,), dtype=float), np.empty((0,), dtype=int)

        return (
            np.asarray(point_coords, dtype=float),
            np.asarray(point_states, dtype=np.int8),
            np.asarray(point_metrics, dtype=float),
            np.asarray(point_rows, dtype=int),
        )

    def _get_smif_profile_process_rows(self, profile=None):
        source_profile = profile if isinstance(profile, dict) else None
        if not isinstance(source_profile, dict):
            try:
                source_profile = self._get_saved_kc_profile_for_input()
            except Exception:
                source_profile = getattr(self, "active_kc_profile", None)
        if not isinstance(source_profile, dict):
            return []

        process_path = str(source_profile.get("process_path") or "").strip()
        if not process_path or not os.path.exists(process_path):
            return []
        if not getattr(self, "gcode_profile", None):
            return []
        if not hasattr(self, "process_single_file"):
            return []

        cache_key = (
            self._normalize_profile_binding_path(process_path),
            str(source_profile.get("updated_at") or ""),
            self._normalize_profile_binding_path(source_profile.get("gcode_path")),
        )
        cached = getattr(self, "_smif_profile_process_rows_cache", None)
        if isinstance(cached, dict) and cached.get("key") == cache_key:
            cached_rows = cached.get("rows") or []
            return [dict(row) for row in cached_rows if isinstance(row, dict)]

        saved_data = list(getattr(self, "data", []) or [])
        saved_map = dict(getattr(self, "raw_to_aligned_line_map", {}) or {})
        saved_lookup = getattr(self, "_process_point_lookup_cache", None)
        saved_lookup_key = getattr(self, "_process_point_lookup_cache_key", None)
        saved_sample_loaded = bool(getattr(self, "sample_data_loaded", False))
        saved_processed_file_path = str(getattr(self, "processed_file_path", "") or "")

        try:
            self.data = []
            self.raw_to_aligned_line_map = {}
            self._process_point_lookup_cache = None
            self._process_point_lookup_cache_key = None
            self.sample_data_loaded = False
            self.process_single_file(process_path)
            if hasattr(self, "_ensure_process_point_metadata"):
                try:
                    self._ensure_process_point_metadata()
                except Exception:
                    pass
            rows = [dict(row) for row in (self.data or []) if isinstance(row, dict)]
            if rows:
                self._smif_profile_process_rows_cache = {
                    "key": cache_key,
                    "rows": rows,
                }
                return [dict(row) for row in rows]
        except Exception:
            return []
        finally:
            self.data = saved_data
            self.raw_to_aligned_line_map = saved_map
            self._process_point_lookup_cache = saved_lookup
            self._process_point_lookup_cache_key = saved_lookup_key
            self.sample_data_loaded = saved_sample_loaded
            self.processed_file_path = saved_processed_file_path

        return []

    def _build_smif_source_payload_from_profile_only(self, metric_key, display_segments, interval_records, focus_bounds, fallback_metric, beta=0.0, global_sigma=0.0):
        def _resolve_record_path_span(record):
            if not isinstance(record, dict):
                return None
            candidates = (
                ("start_s", "end_s"),
                ("path_start", "path_end"),
            )
            for start_key, end_key in candidates:
                try:
                    start_value = float(record.get(start_key))
                    end_value = float(record.get(end_key))
                except Exception:
                    continue
                if np.isfinite(start_value) and np.isfinite(end_value):
                    return (
                        float(min(start_value, end_value)),
                        float(max(start_value, end_value)),
                    )
            return None

        point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})
        line_kc_map = {}
        segment_records = self._get_current_segment_records(allow_profile_fallback=False)
        if not point_kc_map and not line_kc_map and not interval_records and not segment_records:
            self._smif_source_cache = None
            return None

        point_values_by_line = {}
        for point_key, kc_value in (point_kc_map or {}).items():
            try:
                line_no = int(point_key[0])
                point_idx = int(point_key[1])
                metric_value = float(kc_value)
            except Exception:
                continue
            if point_idx < 0 or not np.isfinite(metric_value):
                continue
            point_values_by_line.setdefault(int(line_no), []).append((int(point_idx), max(float(metric_value), 0.0)))
        for line_no in list(point_values_by_line.keys()):
            point_values_by_line[line_no] = sorted(point_values_by_line[line_no], key=lambda item: item[0])

        interval_infos = []
        seen_interval_keys = set()
        source_records = list(interval_records or []) + segment_records
        for interval_index, record in enumerate(source_records):
            if not isinstance(record, dict):
                continue
            line_span = self._resolve_smif_interval_line_span(record)
            if line_span is None:
                continue
            state_code = int(self._resolve_smif_state_code(record))
            dedupe_key = (int(line_span[0]), int(line_span[1]), int(state_code))
            if dedupe_key in seen_interval_keys:
                continue
            seen_interval_keys.add(dedupe_key)
            try:
                metric_raw = float(record.get(metric_key))
            except Exception:
                metric_raw = float("nan")
            if np.isfinite(metric_raw):
                metric_value = float(metric_raw)
            elif int(state_code) == 2:
                metric_value = float(fallback_metric)
            else:
                metric_value = float("nan")
            interval_infos.append({
                "state": int(state_code),
                "metric_value": float(metric_value),
                "line_span": (int(line_span[0]), int(line_span[1])),
                "path_span": _resolve_record_path_span(record),
                "key": str(
                    record.get("segment_id")
                    or record.get("zone_id")
                    or record.get("interval_id")
                    or f"profile_{interval_index + 1}"
                ),
            })

        def _interval_priority(info):
            state_code = int(info.get("state", 1))
            if state_code == 0:
                return 3
            if state_code == 2:
                return 2
            return 1

        def _resolve_profile_segment_state(segment, default_state):
            try:
                seg_line = int(segment.get("n_value"))
            except Exception:
                seg_line = None
            try:
                seg_path_start = float(segment.get("path_start"))
                seg_path_end = float(segment.get("path_end"))
                if np.isfinite(seg_path_start) and np.isfinite(seg_path_end):
                    seg_path_span = (
                        float(min(seg_path_start, seg_path_end)),
                        float(max(seg_path_start, seg_path_end)),
                    )
                else:
                    seg_path_span = None
            except Exception:
                seg_path_span = None
            if seg_line is None:
                if seg_path_span is None:
                    return {
                        "state": int(default_state),
                        "metric_value": float("nan"),
                        "key": "idle" if int(default_state) == 1 else "nonsteady",
                    }
            best_info = None
            best_priority = -1
            best_span = float("inf")
            best_overlap = -1.0
            if seg_path_span is not None:
                for info in interval_infos:
                    path_span = info.get("path_span")
                    if path_span is None:
                        continue
                    overlap_start = max(float(seg_path_span[0]), float(path_span[0]))
                    overlap_end = min(float(seg_path_span[1]), float(path_span[1]))
                    if overlap_end < overlap_start - 1e-9:
                        continue
                    priority = _interval_priority(info)
                    overlap_size = max(float(overlap_end - overlap_start), 0.0)
                    span_size = max(float(path_span[1]) - float(path_span[0]), 1e-9)
                    if (
                        priority > best_priority
                        or (priority == best_priority and overlap_size > best_overlap + 1e-9)
                        or (
                            priority == best_priority
                            and abs(overlap_size - best_overlap) <= 1e-9
                            and span_size < best_span
                        )
                    ):
                        best_info = info
                        best_priority = priority
                        best_span = span_size
                        best_overlap = overlap_size
            if best_info is None and seg_line is not None:
                for info in interval_infos:
                    line_span = info.get("line_span")
                    if line_span is None or not (int(line_span[0]) <= int(seg_line) <= int(line_span[1])):
                        continue
                    priority = _interval_priority(info)
                    span = max(int(line_span[1]) - int(line_span[0]), 0)
                    if priority > best_priority or (priority == best_priority and span < best_span):
                        best_info = info
                        best_priority = priority
                        best_span = span
            if best_info is not None:
                return dict(best_info)
            has_profile_metric = bool(point_values_by_line.get(int(seg_line))) or (
                isinstance(line_kc_map, dict) and int(seg_line) in line_kc_map
            ) if seg_line is not None else False
            if has_profile_metric and int(default_state) != 1:
                return {
                    "state": 2,
                    "metric_value": float("nan"),
                    "key": "steady",
                }
            return {
                "state": int(default_state),
                "metric_value": float("nan"),
                "key": "idle" if int(default_state) == 1 else "nonsteady",
            }

        merged_blocks = []
        all_coords = []
        point_coords = []
        point_states = []
        point_metrics = []
        point_rows = []
        current_block = None

        def _flush_block():
            nonlocal current_block
            if not current_block:
                return
            coords_arr = np.asarray(current_block["coords"], dtype=float)
            if coords_arr.ndim == 2 and coords_arr.shape[1] == 3 and coords_arr.size > 0:
                merged_blocks.append({
                    "state": int(current_block["state"]),
                    "coords": coords_arr,
                    "metrics": np.asarray(current_block["metrics"], dtype=float),
                    "row_indices": np.asarray(current_block["row_indices"], dtype=int),
                })
            current_block = None

        for segment_index, segment in enumerate(display_segments):
            try:
                segment_coords = np.asarray(
                    [
                        [float(segment["start_x"]), float(segment["start_y"]), float(segment["start_z"])],
                        [float(segment["end_x"]), float(segment["end_y"]), float(segment["end_z"])],
                    ],
                    dtype=float,
                )
            except Exception:
                continue
            if segment_coords.shape != (2, 3) or not np.all(np.isfinite(segment_coords)):
                continue

            all_coords.extend([segment_coords[0], segment_coords[1]])
            default_state = int(self._resolve_smif_segment_default_state(segment))
            segment_info = _resolve_profile_segment_state(segment, default_state)
            state_code = int(segment_info.get("state", default_state))
            block_key = str(segment_info.get("key") or ("idle" if state_code == 1 else "nonsteady"))

            try:
                seg_line = int(segment.get("n_value"))
            except Exception:
                seg_line = int(segment_index)
            line_point_values = list(point_values_by_line.get(int(seg_line), []))
            if not line_point_values:
                line_metric = line_kc_map.get(int(seg_line)) if isinstance(line_kc_map, dict) else None
                try:
                    line_metric = float(line_metric)
                except Exception:
                    line_metric = None
                if line_metric is not None and np.isfinite(line_metric):
                    line_point_values = [(0, max(float(line_metric), 0.0))]

            segment_metric = float("nan")
            if state_code != 1 and line_point_values:
                segment_metric = float(np.median(np.asarray([item[1] for item in line_point_values], dtype=float)))
            elif state_code == 2:
                try:
                    segment_metric = float(segment_info.get("metric_value"))
                except Exception:
                    segment_metric = float("nan")
            elif state_code == 0 and line_point_values:
                segment_metric = float(np.median(np.asarray([item[1] for item in line_point_values], dtype=float)))

            can_merge = (
                current_block is not None
                and int(current_block["state"]) == int(state_code)
                and str(current_block["key"]) == str(block_key)
                and np.allclose(
                    np.asarray(current_block["coords"][-1], dtype=float),
                    np.asarray(segment_coords[0], dtype=float),
                    atol=1e-9,
                    rtol=0.0,
                )
            )
            if can_merge:
                current_block["coords"].append(tuple(segment_coords[1]))
                current_block["metrics"].append(float(segment_metric))
                current_block["row_indices"].append(int(seg_line))
            else:
                _flush_block()
                current_block = {
                    "state": int(state_code),
                    "key": str(block_key),
                    "coords": [tuple(segment_coords[0]), tuple(segment_coords[1])],
                    "metrics": [float(segment_metric), float(segment_metric)],
                    "row_indices": [int(seg_line), int(seg_line)],
                }

            if state_code != 1 and line_point_values:
                point_count = max(len(line_point_values), 1)
                segment_delta = segment_coords[1] - segment_coords[0]
                for order_index, (_point_idx, metric_value) in enumerate(line_point_values):
                    interp_t = (float(order_index) + 0.5) / float(point_count)
                    coord = segment_coords[0] + segment_delta * interp_t
                    if not np.all(np.isfinite(coord)):
                        continue
                    point_coords.append(np.asarray(coord, dtype=float))
                    point_states.append(int(state_code))
                    point_metrics.append(float(metric_value))
                    point_rows.append(int(seg_line))

        _flush_block()
        if not merged_blocks and not point_coords:
            self._smif_source_cache = None
            return None

        if all_coords:
            all_coords_arr = np.asarray(all_coords, dtype=float)
            bounds = (np.min(all_coords_arr, axis=0), np.max(all_coords_arr, axis=0))
        else:
            point_coord_arr = np.asarray(point_coords, dtype=float)
            bounds = (np.min(point_coord_arr, axis=0), np.max(point_coord_arr, axis=0))

        finite_metric_values = np.asarray([value for value in point_metrics if np.isfinite(value)], dtype=float)
        if finite_metric_values.size == 0:
            block_metric_values = []
            for block in merged_blocks:
                metrics = np.asarray(block.get("metrics"), dtype=float)
                block_metric_values.extend(float(value) for value in metrics[np.isfinite(metrics)])
            finite_metric_values = np.asarray(block_metric_values, dtype=float)
        if finite_metric_values.size > 0:
            vmin = float(np.min(finite_metric_values))
            vmax = float(np.max(finite_metric_values))
        else:
            vmin = float(fallback_metric)
            vmax = float(fallback_metric + 1.0)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        self._smif_base_projection_payload = {
            "blocks": [np.asarray(block.get("coords"), dtype=float) for block in merged_blocks],
            "focus_bounds": focus_bounds,
            "raw_segment_count": int(len(display_segments)),
        }
        payload = {
            "metric_key": str(metric_key),
            "blocks": merged_blocks,
            "raw_point_count": int(max(
                sum(len(np.asarray(block.get("coords"), dtype=float)) for block in merged_blocks),
                len(point_coords),
            )),
            "bounds": bounds,
            "interval_records": interval_records,
            "point_coords": np.asarray(point_coords, dtype=float),
            "point_states": np.asarray(point_states, dtype=np.int8),
            "point_metrics": np.asarray(point_metrics, dtype=float),
            "point_rows": np.asarray(point_rows, dtype=int),
            "vmin": float(vmin),
            "vmax": float(vmax),
            "cmap_name": "turbo",
            "focus_bounds": focus_bounds or bounds,
            "debug": {"skipped_records": 0},
        }
        self._smif_source_cache = payload
        return payload

    def _build_smif_source_payload(self, metric_key):
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        segment_records = self._get_current_segment_records(allow_profile_fallback=False)
        point_kc_map = dict(getattr(self, "current_interval_point_kc_map", {}) or {})

        profile = getattr(self, "gcode_profile", None)
        segments = list((profile or {}).get("trajectory_segments") or [])
        if not segments:
            self._smif_source_cache = None
            return None

        display_segments = self._get_smif_display_segments(segments)
        if not display_segments:
            display_segments = segments
        focus_segments, focus_bounds = self._resolve_smif_trajectory_focus_segments(display_segments)
        if focus_bounds is not None:
            self._smif_focus_bounds = focus_bounds
        if self._get_smif_view_mode() == "focus" and focus_segments:
            display_segments = focus_segments
        if not display_segments:
            self._smif_source_cache = None
            return None

        beta = float(self.kc_beta.get()) if hasattr(self, "kc_beta") else 0.0
        global_kc = float(self.get_kc_value())
        global_sigma = max(float(self.kc_sigma.get()) if hasattr(self, "kc_sigma") else 0.0, 0.0)
        fallback_metric = float(global_kc + beta * global_sigma) if metric_key == "K_c_UCB" else float(global_kc)

        process_rows = list(getattr(self, "data", []) or [])
        if not process_rows:
            process_rows = self._get_smif_profile_process_rows()
        if not process_rows:
            self._debug_interval_state_event(
                "smif_process_rows_missing",
                current_source=str(getattr(self, "_current_interval_source", "") or "none"),
                interval_count=len(interval_records),
                segment_count=len(segment_records),
            )
            return self._build_smif_source_payload_from_profile_only(
                metric_key,
                display_segments,
                interval_records,
                focus_bounds,
                fallback_metric,
                beta=beta,
                global_sigma=global_sigma,
            )

        total_rows = int(len(process_rows))
        row_indices = np.arange(total_rows, dtype=int)
        row_state_codes = np.ones(total_rows, dtype=np.int8)
        row_interval_metrics = np.full(total_rows, np.nan, dtype=float)
        source_records = segment_records if segment_records else interval_records
        skipped_records = 0
        for interval_index, record in enumerate(source_records):
            if not isinstance(record, dict):
                continue
            state_code = int(self._resolve_smif_state_code(record))
            is_idle_interval = int(state_code) == 1
            try:
                metric_raw = float(record.get(metric_key))
            except Exception:
                metric_raw = float("nan")
            metric_value = float(metric_raw) if np.isfinite(metric_raw) else float(fallback_metric)
            try:
                start_idx = int(record.get("start_idx"))
                end_idx = int(record.get("end_idx"))
            except Exception:
                process_bounds = self._resolve_interval_process_bounds(record, process_rows=process_rows)
                if process_bounds:
                    try:
                        start_idx = int(process_bounds.get("start_idx"))
                        end_idx = int(process_bounds.get("end_idx"))
                    except Exception:
                        start_idx = None
                        end_idx = None
                else:
                    start_idx = None
                    end_idx = None
            if start_idx is None or end_idx is None:
                record_id = str(
                    record.get("segment_id")
                    or record.get("zone_id")
                    or record.get("interval_id")
                    or f"record_{interval_index + 1}"
                )
                self._debug_interval_state_event(
                    "smif_skip_missing_process_bounds",
                    record_id=record_id,
                    current_source=str(getattr(self, "_current_interval_source", "") or "none"),
                )
                skipped_records += 1
                continue
            if end_idx < start_idx:
                start_idx, end_idx = end_idx, start_idx
            safe_start = max(0, min(int(start_idx), total_rows - 1))
            safe_end = max(0, min(int(end_idx), total_rows - 1))
            if safe_end < safe_start:
                record_id = str(
                    record.get("segment_id")
                    or record.get("zone_id")
                    or record.get("interval_id")
                    or f"record_{interval_index + 1}"
                )
                self._debug_interval_state_event(
                    "smif_skip_invalid_process_bounds",
                    record_id=record_id,
                    start_idx=start_idx,
                    end_idx=end_idx,
                )
                skipped_records += 1
                continue
            row_state_codes[safe_start:safe_end + 1] = int(state_code)
            if int(state_code) == 2:
                row_interval_metrics[safe_start:safe_end + 1] = float(metric_value)

        for row_idx, row in enumerate(process_rows):
            if row_state_codes[row_idx] == 2:
                continue
            row_state_codes[row_idx] = 0 if self._is_smif_process_row_cutting(row) else 1

        row_point_metrics = np.full(total_rows, np.nan, dtype=float)
        for row_idx, row in enumerate(process_rows):
            if int(row_state_codes[row_idx]) == 1:
                continue
            metric_value = self._resolve_smif_row_metric_value(
                row,
                row_idx,
                metric_key,
                point_kc_map,
                fallback_metric,
                beta=beta,
                global_sigma=global_sigma,
            )
            if not np.isfinite(metric_value) and int(row_state_codes[row_idx]) == 2:
                metric_value = float(row_interval_metrics[row_idx])
            if np.isfinite(metric_value):
                row_point_metrics[row_idx] = float(metric_value)

        process_block_infos = []
        process_blocks = self._collect_smif_path_blocks(row_indices, row_state_codes)
        for block_start, block_end, block_state in process_blocks:
            block_rows = process_rows[block_start:block_end + 1]
            if not block_rows:
                continue

            path_values = []
            line_values = []
            for row in block_rows:
                for key in ("path_start", "path_end"):
                    try:
                        numeric = float(row.get(key))
                    except Exception:
                        continue
                    if np.isfinite(numeric):
                        path_values.append(float(numeric))
                try:
                    line_values.append(int(row.get("line_no_aligned")))
                except Exception:
                    try:
                        line_values.append(int(row.get("line_no_raw")))
                    except Exception:
                        pass

            if not path_values:
                continue
            block_key = f"state_{int(block_state)}_{int(block_start)}_{int(block_end)}"
            block_metric_value = float("nan")
            if int(block_state) != 1:
                finite_metrics = row_point_metrics[block_start:block_end + 1]
                finite_metrics = finite_metrics[np.isfinite(finite_metrics)]
                if finite_metrics.size > 0:
                    block_metric_value = float(np.nanmedian(finite_metrics))
                elif int(block_state) == 2:
                    interval_metrics = row_interval_metrics[block_start:block_end + 1]
                    interval_metrics = interval_metrics[np.isfinite(interval_metrics)]
                    if interval_metrics.size > 0:
                        block_metric_value = float(np.nanmedian(interval_metrics))
            process_block_infos.append({
                "state": int(block_state),
                "metric_value": float(block_metric_value),
                "path_span": (float(min(path_values)), float(max(path_values))),
                "line_span": (int(min(line_values)), int(max(line_values))) if line_values else None,
                "key": block_key,
            })

        if not process_block_infos:
            self._smif_source_cache = None
            return None
        process_block_infos.sort(key=lambda info: (
            float((info.get("path_span") or (float("inf"), float("inf")))[0]),
            float((info.get("path_span") or (float("inf"), float("inf")))[1]),
            int(info.get("state", 0)),
        ))

        def _segment_tag_value(segment, fallback_value):
            for key in ("n_value", "file_line_index"):
                try:
                    return int(segment.get(key))
                except Exception:
                    continue
            return int(fallback_value)

        def _interval_priority(info):
            try:
                state_code = int(info.get("state"))
            except Exception:
                state_code = 1
            if state_code == 2:
                return 2
            if state_code == 0:
                return 1
            return 0

        def _resolve_segment_subspans(segment):
            try:
                seg_path_start = float(segment.get("path_start"))
                seg_path_end = float(segment.get("path_end"))
                if np.isfinite(seg_path_start) and np.isfinite(seg_path_end):
                    seg_path_span = (min(seg_path_start, seg_path_end), max(seg_path_start, seg_path_end))
                else:
                    seg_path_span = None
            except Exception:
                seg_path_span = None
            try:
                seg_line = int(segment.get("n_value"))
            except Exception:
                seg_line = None

            default_state = int(self._resolve_smif_segment_default_state(segment))
            default_key = "idle" if default_state == 1 else "nonsteady"
            default_piece = {
                "state": int(default_state),
                "metric_value": float("nan"),
                "key": str(default_key),
            }

            if seg_path_span is not None and abs(seg_path_span[1] - seg_path_span[0]) > 1e-9:
                breakpoints = {float(seg_path_span[0]), float(seg_path_span[1])}
                clipped_infos = []
                for info in process_block_infos:
                    path_span = info.get("path_span")
                    if path_span is None:
                        continue
                    overlap_start = max(float(seg_path_span[0]), float(path_span[0]))
                    overlap_end = min(float(seg_path_span[1]), float(path_span[1]))
                    if overlap_end - overlap_start <= 1e-9:
                        continue
                    breakpoints.add(float(overlap_start))
                    breakpoints.add(float(overlap_end))
                    clipped_infos.append((float(overlap_start), float(overlap_end), info))

                if clipped_infos:
                    path_points = sorted(float(point) for point in breakpoints)
                    pieces = []
                    for left, right in zip(path_points[:-1], path_points[1:]):
                        if right - left <= 1e-9:
                            continue
                        midpoint = 0.5 * (left + right)
                        chosen_info = None
                        chosen_priority = -1
                        chosen_span = float("inf")
                        for overlap_start, overlap_end, info in clipped_infos:
                            if midpoint < overlap_start - 1e-9 or midpoint > overlap_end + 1e-9:
                                continue
                            priority = _interval_priority(info)
                            path_span = info.get("path_span")
                            span_size = (
                                max(float(path_span[1]) - float(path_span[0]), 1e-9)
                                if path_span is not None else float("inf")
                            )
                            if priority > chosen_priority or (
                                priority == chosen_priority and span_size < chosen_span
                            ):
                                chosen_info = info
                                chosen_priority = priority
                                chosen_span = span_size
                        if chosen_info is None:
                            piece = dict(default_piece)
                        else:
                            piece = {
                                "state": int(chosen_info.get("state", default_state)),
                                "metric_value": float(chosen_info.get("metric_value", float("nan"))),
                                "key": str(chosen_info.get("key") or default_key),
                            }
                        piece["path_start"] = float(left)
                        piece["path_end"] = float(right)
                        pieces.append(piece)
                    if pieces:
                        return pieces

            line_match = None
            line_priority = -1
            line_span_size = float("inf")
            if seg_line is not None:
                for info in process_block_infos:
                    line_span = info.get("line_span")
                    if line_span is None or not (int(line_span[0]) <= int(seg_line) <= int(line_span[1])):
                        continue
                    priority = _interval_priority(info)
                    span_size = max(int(line_span[1]) - int(line_span[0]), 0)
                    if priority > line_priority or (
                        priority == line_priority and span_size < line_span_size
                    ):
                        line_match = info
                        line_priority = priority
                        line_span_size = span_size

            if line_match is not None:
                piece = {
                    "state": int(line_match.get("state", default_state)),
                    "metric_value": float(line_match.get("metric_value", float("nan"))),
                    "key": str(line_match.get("key") or default_key),
                }
            else:
                piece = dict(default_piece)

            if seg_path_span is not None:
                piece["path_start"] = float(seg_path_span[0])
                piece["path_end"] = float(seg_path_span[1])
            return [piece]

        merged_blocks = []
        all_coords = []
        current_block = None

        def _flush_block():
            nonlocal current_block
            if not current_block:
                return
            coords_arr = np.asarray(current_block["coords"], dtype=float)
            if coords_arr.ndim == 2 and coords_arr.shape[1] == 3 and coords_arr.size > 0:
                merged_blocks.append({
                    "state": int(current_block["state"]),
                    "coords": coords_arr,
                    "metrics": np.asarray(current_block["metrics"], dtype=float),
                    "row_indices": np.asarray(current_block["row_indices"], dtype=int),
                })
            current_block = None

        for segment_index, segment in enumerate(display_segments):
            try:
                segment_coords = np.asarray(
                    [
                        [float(segment["start_x"]), float(segment["start_y"]), float(segment["start_z"])],
                        [float(segment["end_x"]), float(segment["end_y"]), float(segment["end_z"])],
                    ],
                    dtype=float,
                )
            except Exception:
                continue
            if segment_coords.shape != (2, 3) or not np.all(np.isfinite(segment_coords)):
                continue

            tag_value = _segment_tag_value(segment, segment_index)
            try:
                segment_path_start = float(segment.get("path_start"))
                segment_path_end = float(segment.get("path_end"))
                if not (np.isfinite(segment_path_start) and np.isfinite(segment_path_end)):
                    segment_path_start = None
                    segment_path_end = None
            except Exception:
                segment_path_start = None
                segment_path_end = None
            segment_path_delta = (
                float(segment_path_end - segment_path_start)
                if segment_path_start is not None and segment_path_end is not None else None
            )

            subspans = _resolve_segment_subspans(segment)
            if not subspans:
                subspans = [{
                    "state": int(self._resolve_smif_segment_default_state(segment)),
                    "metric_value": float("nan"),
                    "key": "idle" if int(self._resolve_smif_segment_default_state(segment)) == 1 else "nonsteady",
                }]

            for sub_index, subspan in enumerate(subspans):
                if segment_path_delta is not None and abs(segment_path_delta) > 1e-9:
                    sub_path_start = subspan.get("path_start")
                    sub_path_end = subspan.get("path_end")
                    if sub_path_start is None or sub_path_end is None:
                        t_start = 0.0
                        t_end = 1.0
                    else:
                        t_start = (float(sub_path_start) - float(segment_path_start)) / float(segment_path_delta)
                        t_end = (float(sub_path_end) - float(segment_path_start)) / float(segment_path_delta)
                    t_start = float(min(max(t_start, 0.0), 1.0))
                    t_end = float(min(max(t_end, 0.0), 1.0))
                    part_start = segment_coords[0] + (segment_coords[1] - segment_coords[0]) * t_start
                    part_end = segment_coords[0] + (segment_coords[1] - segment_coords[0]) * t_end
                    part_coords = np.asarray([part_start, part_end], dtype=float)
                else:
                    part_coords = np.asarray(segment_coords, dtype=float)

                if (
                    part_coords.shape != (2, 3)
                    or not np.all(np.isfinite(part_coords))
                    or np.allclose(part_coords[0], part_coords[1], atol=1e-9, rtol=0.0)
                ):
                    continue

                all_coords.extend([part_coords[0], part_coords[1]])
                state_code = int(subspan.get("state", 1))
                metric_value = float(subspan.get("metric_value", float("nan"))) if state_code != 1 else float("nan")
                block_key = str(subspan.get("key") or f"state_{state_code}")
                can_merge = (
                    current_block is not None
                    and int(current_block["state"]) == int(state_code)
                    and str(current_block["key"]) == str(block_key)
                    and np.allclose(
                        np.asarray(current_block["coords"][-1], dtype=float),
                        np.asarray(part_coords[0], dtype=float),
                        atol=1e-9,
                        rtol=0.0,
                    )
                )
                if can_merge:
                    current_block["coords"].append(tuple(part_coords[1]))
                    current_block["metrics"].append(float(metric_value))
                    current_block["row_indices"].append(int(tag_value))
                    continue

                _flush_block()
                current_block = {
                    "state": int(state_code),
                    "key": str(block_key),
                    "coords": [tuple(part_coords[0]), tuple(part_coords[1])],
                    "metrics": [float(metric_value), float(metric_value)],
                    "row_indices": [int(tag_value), int(tag_value)],
                }

        _flush_block()
        if not merged_blocks or not all_coords:
            self._smif_source_cache = None
            return None

        all_coords_arr = np.asarray(all_coords, dtype=float)
        point_coords, point_states, point_metrics, point_rows = self._map_smif_process_points_to_segments(
            display_segments,
            process_rows,
            row_state_codes,
            row_point_metrics,
        )

        finite_metric_values = row_point_metrics[np.isfinite(row_point_metrics)]
        if finite_metric_values.size > 0:
            vmin = float(np.min(finite_metric_values))
            vmax = float(np.max(finite_metric_values))
        else:
            vmin = float(fallback_metric)
            vmax = float(fallback_metric + 1.0)
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        self._smif_base_projection_payload = {
            "blocks": [np.asarray(block.get("coords"), dtype=float) for block in merged_blocks],
            "focus_bounds": focus_bounds,
            "raw_segment_count": int(len(display_segments)),
        }
        payload = {
            "metric_key": str(metric_key),
            "blocks": merged_blocks,
            "raw_point_count": int(sum(len(np.asarray(block.get("coords"), dtype=float)) for block in merged_blocks)),
            "bounds": (
                np.min(all_coords_arr, axis=0),
                np.max(all_coords_arr, axis=0),
            ),
            "interval_records": interval_records,
            "point_coords": point_coords,
            "point_states": point_states,
            "point_metrics": point_metrics,
            "point_rows": point_rows,
            "vmin": float(vmin),
            "vmax": float(vmax),
            "cmap_name": "turbo",
            "focus_bounds": focus_bounds or self._compute_smif_effective_bounds(all_coords_arr),
        }
        self._smif_source_cache = payload
        return payload

    def _plot_process_interval_trajectory(self, ax, metric_key, view_limits=None, reuse_source_cache=False):
        payload = None
        if reuse_source_cache:
            cached_payload = getattr(self, "_smif_source_cache", None)
            if isinstance(cached_payload, dict) and str(cached_payload.get("metric_key")) == str(metric_key):
                payload = cached_payload
        if payload is None:
            payload = self._build_smif_source_payload(metric_key)
        if not isinstance(payload, dict):
            self._smif_dashboard_payload = None
            return False

        blocks = list(payload.get("blocks") or [])
        point_coords = np.asarray(payload.get("point_coords", np.empty((0, 3), dtype=float)), dtype=float)
        point_states = np.asarray(payload.get("point_states", np.empty((0,), dtype=np.int8)), dtype=np.int8)
        point_metrics = np.asarray(payload.get("point_metrics", np.empty((0,), dtype=float)), dtype=float)
        point_rows = np.asarray(payload.get("point_rows", np.empty((0,), dtype=int)), dtype=int)
        raw_point_count = max(int(payload.get("raw_point_count", 0) or 0), int(len(point_coords)))
        if not blocks and point_coords.size == 0:
            self._smif_dashboard_payload = None
            return False
        steady_scope_mode = self._get_smif_scope_mode() == "steady"

        payload_bounds = payload.get("bounds")
        if isinstance(payload_bounds, tuple) and len(payload_bounds) == 2:
            try:
                self._update_smif_bounds(np.asarray(payload_bounds, dtype=float))
            except Exception:
                pass
        focus_bounds = payload.get("focus_bounds")
        if isinstance(focus_bounds, tuple) and len(focus_bounds) == 2:
            self._smif_focus_bounds = focus_bounds
        cmap = plt.get_cmap(str(payload.get("cmap_name") or "coolwarm"))
        norm = matplotlib.colors.Normalize(
            vmin=float(payload.get("vmin", 0.0)),
            vmax=float(payload.get("vmax", 1.0)),
        )

        render_blocks = blocks
        visible_point_count = raw_point_count
        render_point_coords = np.asarray(point_coords, dtype=float)
        render_point_states = np.asarray(point_states, dtype=np.int8)
        render_point_metrics = np.asarray(point_metrics, dtype=float)
        render_point_rows = np.asarray(point_rows, dtype=int)
        if isinstance(view_limits, dict):
            try:
                xlim = tuple(sorted(float(v) for v in view_limits.get("xlim", ())))
                ylim = tuple(sorted(float(v) for v in view_limits.get("ylim", ())))
                zlim = tuple(sorted(float(v) for v in view_limits.get("zlim", ())))
            except Exception:
                xlim = ylim = zlim = ()
            if len(xlim) == len(ylim) == len(zlim) == 2:
                x_margin = max(abs(xlim[1] - xlim[0]) * 0.03, 0.05)
                y_margin = max(abs(ylim[1] - ylim[0]) * 0.03, 0.05)
                z_margin = max(abs(zlim[1] - zlim[0]) * 0.03, 0.05)
                filtered_blocks = []
                visible_point_count = 0
                for block in blocks:
                    normalized = self._normalize_smif_dashboard_block(block)
                    if normalized is None:
                        continue
                    block_coords, _metrics, _rows, _state = normalized
                    display_coords = self._transform_smif_coords(block_coords)
                    if display_coords.shape != block_coords.shape or not np.all(np.isfinite(display_coords)):
                        continue
                    seg_x_min = float(np.min(display_coords[:, 0]))
                    seg_x_max = float(np.max(display_coords[:, 0]))
                    seg_y_min = float(np.min(display_coords[:, 1]))
                    seg_y_max = float(np.max(display_coords[:, 1]))
                    seg_z_min = float(np.min(display_coords[:, 2]))
                    seg_z_max = float(np.max(display_coords[:, 2]))
                    intersects = not (
                        seg_x_max < xlim[0] - x_margin or seg_x_min > xlim[1] + x_margin
                        or seg_y_max < ylim[0] - y_margin or seg_y_min > ylim[1] + y_margin
                        or seg_z_max < zlim[0] - z_margin or seg_z_min > zlim[1] + z_margin
                    )
                    if intersects:
                        filtered_blocks.append(block)
                        visible_point_count += int(len(block_coords))
                render_blocks = filtered_blocks

                if render_point_coords.ndim == 2 and render_point_coords.shape[1] == 3 and render_point_coords.size > 0:
                    display_points = self._transform_smif_coords(render_point_coords)
                    if display_points.shape == render_point_coords.shape and np.all(np.isfinite(display_points)):
                        point_mask = (
                            (display_points[:, 0] >= xlim[0] - x_margin)
                            & (display_points[:, 0] <= xlim[1] + x_margin)
                            & (display_points[:, 1] >= ylim[0] - y_margin)
                            & (display_points[:, 1] <= ylim[1] + y_margin)
                            & (display_points[:, 2] >= zlim[0] - z_margin)
                            & (display_points[:, 2] <= zlim[1] + z_margin)
                        )
                        render_point_coords = render_point_coords[point_mask]
                        render_point_states = render_point_states[:len(point_mask)][point_mask[:len(render_point_states)]]
                        render_point_metrics = render_point_metrics[:len(point_mask)][point_mask[:len(render_point_metrics)]]
                        render_point_rows = render_point_rows[:len(point_mask)][point_mask[:len(render_point_rows)]]
                        visible_point_count = max(int(visible_point_count), int(np.sum(point_mask)))

        steady_segments = []
        steady_colors = []
        idle_segments = []
        steady_single_points = []
        steady_single_point_colors = []
        steady_process_points = []
        steady_process_point_colors = []
        idle_single_points = []
        nonsteady_single_points = []
        nonsteady_single_point_colors = []
        nonsteady_process_points = []
        nonsteady_process_point_colors = []
        background_segments = []
        background_single_points = []
        interaction_points = []
        dashboard_blocks = []
        can_draw_3d = self._axis_is_3d(ax)
        metric_present = False
        interval_line_width = float(self._get_smif_interval_line_width())
        interval_point_size = float(self._get_smif_interval_point_size())
        has_nonsteady_point_overlay = bool(
            render_point_coords.ndim == 2
            and render_point_coords.shape[1] == 3
            and render_point_coords.size > 0
            and np.any(render_point_states[:len(render_point_coords)] == 0)
        )
        for block in render_blocks:
            normalized = self._normalize_smif_dashboard_block(block)
            if normalized is None:
                continue
            block_coords, block_metrics, block_rows, block_state = normalized
            if block_coords.size == 0:
                continue

            display_coords = self._transform_smif_coords(block_coords)
            if display_coords.shape != block_coords.shape or not np.all(np.isfinite(display_coords)):
                continue

            interaction_points.append(display_coords)
            dashboard_blocks.append({
                "state": int(block_state),
                "coords": np.asarray(block_coords, dtype=float),
                "metrics": np.asarray(block_metrics, dtype=float),
                "row_indices": np.asarray(block_rows, dtype=int),
            })
            if len(display_coords) == 1:
                if steady_scope_mode:
                    background_single_points.append(tuple(display_coords[0]))
                if int(block_state) == 1:
                    idle_single_points.append(tuple(display_coords[0]))
                elif int(block_state) == 2:
                    finite_metrics = np.asarray(block_metrics[np.isfinite(block_metrics)], dtype=float)
                    metric_value = float(finite_metrics[0]) if finite_metrics.size else float(norm.vmin)
                    steady_single_points.append(tuple(display_coords[0]))
                    steady_single_point_colors.append(cmap(norm(metric_value)))
                    metric_present = True
                else:
                    if has_nonsteady_point_overlay:
                        continue
                    nonsteady_single_points.append(tuple(display_coords[0]))
                    finite_metrics = np.asarray(block_metrics[np.isfinite(block_metrics)], dtype=float)
                    metric_value = float(finite_metrics[0]) if finite_metrics.size else float("nan")
                    nonsteady_single_point_colors.append(
                        cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR
                    )
                    metric_present = metric_present or bool(np.isfinite(metric_value))
                continue

            block_segments = [
                [tuple(display_coords[idx]), tuple(display_coords[idx + 1])]
                for idx in range(len(display_coords) - 1)
            ]
            if steady_scope_mode:
                background_segments.extend(block_segments)
            if int(block_state) == 1:
                idle_segments.extend(block_segments)
            elif int(block_state) == 2:
                finite_metrics = np.asarray(block_metrics[np.isfinite(block_metrics)], dtype=float)
                metric_value = float(np.nanmedian(finite_metrics)) if finite_metrics.size else float(norm.vmin)
                steady_segments.extend(block_segments)
                steady_colors.extend([cmap(norm(metric_value))] * len(block_segments))
                metric_present = True
            else:
                if has_nonsteady_point_overlay:
                    continue
                fallback_metrics = block_metrics if block_metrics.size else np.full(len(display_coords), np.nan, dtype=float)
                for point_idx, coord in enumerate(display_coords):
                    nonsteady_single_points.append(tuple(coord))
                    metric_value = float(fallback_metrics[min(point_idx, len(fallback_metrics) - 1)]) if len(fallback_metrics) else float("nan")
                    nonsteady_single_point_colors.append(
                        cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR
                    )
                    metric_present = metric_present or bool(np.isfinite(metric_value))

        if render_point_coords.ndim == 2 and render_point_coords.shape[1] == 3 and render_point_coords.size > 0:
            display_point_coords = self._transform_smif_coords(render_point_coords)
            if display_point_coords.shape == render_point_coords.shape and np.all(np.isfinite(display_point_coords)):
                interaction_points.append(display_point_coords)
                for idx, coord in enumerate(display_point_coords):
                    state_code = int(render_point_states[idx]) if idx < len(render_point_states) else 1
                    metric_value = float(render_point_metrics[idx]) if idx < len(render_point_metrics) else float("nan")
                    if state_code == 0:
                        nonsteady_process_points.append(tuple(coord))
                        nonsteady_process_point_colors.append(cmap(norm(metric_value)) if np.isfinite(metric_value) else SMIF_NONSTEADY_COLOR)
                        metric_present = metric_present or bool(np.isfinite(metric_value))
                    elif state_code == 2:
                        steady_process_points.append(tuple(coord))
                        steady_color_value = float(metric_value) if np.isfinite(metric_value) else float(norm.vmin)
                        steady_process_point_colors.append(cmap(norm(steady_color_value)))
                        metric_present = True
                    elif state_code != 1:
                        metric_present = metric_present or bool(np.isfinite(metric_value))

        background_line_color = "#C6CBD2"
        background_point_color = "#D2D6DB"
        if can_draw_3d and steady_scope_mode and background_segments:
            background_collection = Line3DCollection(
                background_segments,
                colors=background_line_color,
                linewidths=max(interval_line_width * 0.55, 1.0),
                linestyles="solid",
                capstyle="butt",
                joinstyle="miter",
                alpha=0.42,
            )
            ax.add_collection3d(background_collection)
        if can_draw_3d and steady_scope_mode and background_single_points:
            background_coords = np.asarray(background_single_points, dtype=float)
            ax.scatter(
                background_coords[:, 0],
                background_coords[:, 1],
                background_coords[:, 2],
                c=background_point_color,
                s=max(interval_point_size * 0.42, 4.0),
                marker="o",
                alpha=0.46,
                depthshade=False,
                edgecolors="none",
                zorder=4,
            )

        if can_draw_3d and idle_segments:
            idle_collection = Line3DCollection(
                idle_segments,
                colors=background_line_color if steady_scope_mode else SMIF_IDLE_COLOR,
                linewidths=max(interval_line_width * 0.72, 1.0),
                linestyles="solid",
                capstyle="butt",
                joinstyle="miter",
                alpha=0.38 if steady_scope_mode else 0.82,
            )
            ax.add_collection3d(idle_collection)
        if can_draw_3d and idle_single_points:
            idle_coords = np.asarray(idle_single_points, dtype=float)
            ax.scatter(
                idle_coords[:, 0],
                idle_coords[:, 1],
                idle_coords[:, 2],
                c=background_point_color if steady_scope_mode else SMIF_IDLE_COLOR,
                s=max(interval_point_size * 0.45, 4.0),
                marker="o",
                alpha=0.40 if steady_scope_mode else 0.92,
                depthshade=False,
                edgecolors="none",
                zorder=5,
            )
        if can_draw_3d and steady_segments:
            steady_collection = Line3DCollection(
                steady_segments,
                colors=steady_colors,
                linewidths=interval_line_width,
                linestyles="solid",
                capstyle="butt",
                joinstyle="miter",
                alpha=0.96,
            )
            ax.add_collection3d(steady_collection)
        if can_draw_3d and steady_single_points:
            steady_coords = np.asarray(steady_single_points, dtype=float)
            ax.scatter(
                steady_coords[:, 0],
                steady_coords[:, 1],
                steady_coords[:, 2],
                c=steady_single_point_colors,
                s=interval_point_size,
                marker="o",
                alpha=0.98,
                depthshade=False,
                edgecolors="none",
                zorder=6,
            )
        if can_draw_3d and steady_process_points:
            steady_process_arr = np.asarray(steady_process_points, dtype=float)
            ax.scatter(
                steady_process_arr[:, 0],
                steady_process_arr[:, 1],
                steady_process_arr[:, 2],
                c=steady_process_point_colors,
                s=max(interval_point_size * 0.46, 3.2),
                marker="o",
                alpha=0.97,
                depthshade=False,
                edgecolors="none",
                zorder=7,
            )
        nonsteady_point_size = max(interval_point_size * 0.40, 2.4)
        if can_draw_3d and nonsteady_process_points:
            nonsteady_point_arr = np.asarray(nonsteady_process_points, dtype=float)
            ax.scatter(
                nonsteady_point_arr[:, 0],
                nonsteady_point_arr[:, 1],
                nonsteady_point_arr[:, 2],
                c=background_point_color if steady_scope_mode else nonsteady_process_point_colors,
                s=nonsteady_point_size,
                marker="o",
                alpha=0.42 if steady_scope_mode else 0.94,
                depthshade=False,
                edgecolors="none",
                zorder=8,
            )
        if can_draw_3d and nonsteady_single_points:
            nonsteady_coords = np.asarray(nonsteady_single_points, dtype=float)
            ax.scatter(
                nonsteady_coords[:, 0],
                nonsteady_coords[:, 1],
                nonsteady_coords[:, 2],
                c=background_point_color if steady_scope_mode else nonsteady_single_point_colors,
                s=nonsteady_point_size,
                marker="o",
                alpha=0.42 if steady_scope_mode else 0.94,
                depthshade=False,
                edgecolors="none",
                zorder=8,
            )

        if interaction_points:
            merged_interaction = np.vstack(interaction_points)
            zoom_target = min(len(merged_interaction), max(1000, min(3000, max(visible_point_count, 1000) // 2)))
            zoom_indices = self._compress_smif_path_indices(merged_interaction, zoom_target)
            self._smif_interaction_points = merged_interaction[zoom_indices]
        else:
            self._smif_interaction_points = np.empty((0, 3), dtype=float)

        line_display_point_count = int(sum(len(np.asarray(block.get("coords"), dtype=float)) for block in render_blocks))
        point_display_count = int(len(render_point_coords)) if render_point_coords.ndim == 2 else 0
        display_point_count = max(line_display_point_count, point_display_count)
        self._smif_render_stats = {
            "raw_point_count": raw_point_count,
            "visible_point_count": int(visible_point_count),
            "display_target": int(max(display_point_count, visible_point_count)),
            "display_point_count": int(display_point_count),
            "line_budget": int(display_point_count),
            "nonsteady_budget": int(len(nonsteady_process_points) + len(nonsteady_single_points)),
            "simplified": bool(raw_point_count > visible_point_count and visible_point_count > 0),
        }
        self._smif_render_stats["annotation_count"] = 0
        self._smif_dashboard_payload = {
            "metric_key": str(metric_key),
            "metric_range": (float(norm.vmin), float(norm.vmax)),
            "cmap_name": str(payload.get("cmap_name") or "turbo"),
            "blocks": dashboard_blocks,
            "point_coords": np.asarray(render_point_coords, dtype=float),
            "point_states": np.asarray(render_point_states, dtype=np.int8),
            "point_metrics": np.asarray(render_point_metrics, dtype=float),
            "point_rows": np.asarray(render_point_rows, dtype=int),
            "interval_records": list(payload.get("interval_records") or []),
        }

        colorbar_ax = getattr(self, "_smif_colorbar_ax", None)
        if colorbar_ax is not None:
            try:
                colorbar_ax.cla()
                colorbar_ax.set_visible(False)
                colorbar_ax.set_facecolor(SMIF_FIG_BG)
            except Exception:
                pass
        if metric_present and colorbar_ax is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            colorbar_ax.set_visible(True)
            colorbar = self.fig_smif.colorbar(sm, cax=colorbar_ax)
            label = self._get_smif_metric_label(metric_key)
            colorbar.set_label(label, color=SMIF_TEXT_COLOR)
            colorbar.ax.yaxis.label.set_color(SMIF_TEXT_COLOR)
            colorbar.outline.set_edgecolor(SMIF_PANE_EDGE)
            colorbar_ax.tick_params(labelsize=max(PLOT_FONT_BASE - 2, 8), colors=SMIF_TEXT_COLOR)
        return bool(
            blocks
            or (render_point_coords.ndim == 2 and render_point_coords.shape[0] > 0)
        )

    def refresh_smif_view(self, view_state=None, reuse_source_cache=False):
        if not hasattr(self, "ax_smif") or not hasattr(self, "canvas_smif"):
            return

        if not self._has_smif_trajectory_source():
            self._cancel_pending_smif_view_refresh()
            self._reset_smif_runtime_cache()
            self._draw_smif_empty_placeholder()
            try:
                self.canvas_smif.draw()
            except Exception as exc:
                if hasattr(self, "_report_view_refresh_error"):
                    self._report_view_refresh_error("SMIF占位图绘制", exc)
                self.canvas_smif.draw_idle()
            self._reload_smif_pit_tree()
            if hasattr(self, "smif_status_var"):
                if getattr(self, "data", None):
                    self.smif_status_var.set("未导入G代码NC：SMIF 3D轨迹待导入，PIT/区间结果仍可使用")
                else:
                    self.smif_status_var.set("未导入G代码NC（工艺信息分析可直接使用输入中的 s/S）")
            return

        self._cancel_pending_smif_view_refresh()
        self._reset_smif_runtime_cache()
        # 在创建 axes 之前，先将 figure 尺寸同步到画布实际大小
        if hasattr(self, '_sync_smif_figure_to_canvas'):
            try:
                self._sync_smif_figure_to_canvas()
            except Exception:
                pass
        self._build_smif_dashboard_axes()

        effective_view = view_state if isinstance(view_state, dict) else None
        metric_var = getattr(self, "smif_metric_var", None)
        metric_key = str(metric_var.get()).strip() if metric_var is not None else "K_c_hat"
        if metric_key not in {"K_c_hat", "K_c_UCB"}:
            metric_key = "K_c_hat"
        has_colored_intervals = self._plot_process_interval_trajectory(
            self.ax_smif,
            metric_key,
            view_limits=effective_view,
            reuse_source_cache=bool(reuse_source_cache),
        )
        scope_mode = self._get_smif_scope_mode()
        max_base_segments = 8000 if not self.data else 4000
        has_base_trajectory = False
        if not has_colored_intervals and scope_mode == "all":
            has_base_trajectory = self._plot_nc_trajectory(self.ax_smif, max_segments=max_base_segments, view_limits=effective_view)

        if has_base_trajectory or has_colored_intervals:
            if effective_view:
                self._apply_smif_view_state(self.ax_smif, effective_view)
            else:
                try:
                    self._apply_smif_axis_limits(self.ax_smif)
                    self._auto_adjust_smif_view_angle(self.ax_smif)
                except Exception:
                    pass
            self._render_smif_side_panels(metric_key, has_colored_intervals=bool(has_colored_intervals))
            self._draw_smif_summary_box(metric_key, has_colored_intervals=bool(has_colored_intervals))
        else:
            self.ax_smif.text2D(
                0.5, 0.5,
                "当前范围内暂无可显示轨迹\n可切回“全部显示”或先生成稳态区间",
                ha="center", va="center", transform=self.ax_smif.transAxes,
                fontsize=PLOT_FONT_BASE + 1, color=SMIF_TEXT_MUTED
            )
            self.ax_smif.set_xticks([])
            self.ax_smif.set_yticks([])
            self.ax_smif.set_zticks([])
            self._render_smif_side_panels(metric_key, has_colored_intervals=False)

        try:
            self.canvas_smif.draw()
        except Exception as exc:
            if hasattr(self, "_report_view_refresh_error"):
                self._report_view_refresh_error("SMIF主视图绘制", exc)
            self.canvas_smif.draw_idle()
        self._reload_smif_pit_tree()

        if hasattr(self, "smif_status_var"):
            scope_text = "仅稳态" if scope_mode == "steady" else "全部显示"
            if has_colored_intervals:
                stats = getattr(self, "_smif_render_stats", {}) or {}
                raw_point_count = int(stats.get("raw_point_count", 0) or 0)
                visible_point_count = int(stats.get("visible_point_count", raw_point_count) or raw_point_count)
                display_point_count = int(stats.get("display_point_count", raw_point_count) or raw_point_count)
                if bool(stats.get("simplified", False)) and raw_point_count > 0:
                    self.smif_status_var.set(
                        f"SMIF已更新: {metric_key}，{scope_text}，总 {raw_point_count} 点，可见 {visible_point_count} 点，显示 {display_point_count} 点(自适应抽稀)，稳态区间 {len(self._get_current_interval_records(allow_profile_fallback=False))} 段"
                    )
                else:
                    self.smif_status_var.set(
                        f"SMIF已更新: {metric_key}，{scope_text}，显示 {display_point_count} 个轨迹点，稳态区间 {len(self._get_current_interval_records(allow_profile_fallback=False))} 段"
                    )
            elif has_base_trajectory:
                self.smif_status_var.set("NC轨迹已生成，等待区间 Kc 更新")
            elif scope_mode == "steady":
                self.smif_status_var.set("SMIF已更新: 仅稳态，当前没有可显示的稳态轨迹")
            else:
                self.smif_status_var.set("未导入G代码NC（工艺信息分析可直接使用输入中的 s/S）")

    def on_smif_scroll_zoom(self, event):
        ax = getattr(self, "ax_smif", None)
        canvas = getattr(self, "canvas_smif", None)
        if ax is None or canvas is None:
            return
        if not all(hasattr(ax, name) for name in ("get_xlim3d", "get_ylim3d", "get_zlim3d")):
            return
        if event.inaxes is not ax:
            return

        if event.button == "up":
            scale = 1.0 / float(self.zoom_factor)
        elif event.button == "down":
            scale = float(self.zoom_factor)
        else:
            return

        current_limits = [
            tuple(float(v) for v in ax.get_xlim3d()),
            tuple(float(v) for v in ax.get_ylim3d()),
            tuple(float(v) for v in ax.get_zlim3d()),
        ]
        anchor = self._resolve_smif_zoom_anchor(ax, event)
        if anchor is None:
            anchor = tuple((axis_min + axis_max) / 2.0 for axis_min, axis_max in current_limits)

        min_zoom_range = 1e-6
        for axis_index, (axis_min, axis_max) in enumerate(current_limits):
            axis_range = max(float(axis_max - axis_min), min_zoom_range)
            anchor_value = float(anchor[axis_index])
            ratio = (anchor_value - axis_min) / axis_range if axis_range > 0 else 0.5
            ratio = min(max(ratio, 0.0), 1.0)
            new_range = max(axis_range * scale, min_zoom_range)
            new_min = anchor_value - ratio * new_range
            new_max = new_min + new_range
            if axis_index == 0:
                ax.set_xlim3d(new_min, new_max)
            elif axis_index == 1:
                ax.set_ylim3d(new_min, new_max)
            else:
                ax.set_zlim3d(new_min, new_max)
        try:
            self._set_smif_viewer_box(ax)
        except Exception:
            pass
        ax.set_anchor('C')
        self._refresh_smif_main_axis_ticks(ax)
        canvas.draw_idle()

    def _smif_event_has_alt(self, event):
        modifiers = getattr(event, "modifiers", None)
        if modifiers is not None:
            modifier_tokens = {
                str(token or "").strip().lower()
                for token in modifiers
                if str(token or "").strip()
            }
            return ("alt" in modifier_tokens) or ("altgr" in modifier_tokens)

        if hasattr(event, "key"):
            key_value = getattr(event, "key", None)
            if key_value is not None:
                return "alt" in str(key_value or "").strip().lower()

        gui_event = getattr(event, "guiEvent", None)
        state_value = getattr(gui_event, "state", None)
        if isinstance(state_value, int):
            alt_masks = (0x0008, 0x0080, 0x20000)
            if any(int(state_value) & mask for mask in alt_masks):
                return True
            if hasattr(event, "button"):
                return False

        if hasattr(event, "button"):
            return False
        return bool(getattr(self, "_smif_alt_pressed", False))

    def on_smif_key_press(self, event):
        key_text = str(getattr(event, "key", "") or "").strip().lower()
        if "alt" in key_text:
            self._smif_alt_pressed = True

    def on_smif_key_release(self, event):
        key_text = str(getattr(event, "key", "") or "").strip().lower()
        if "alt" in key_text:
            self._smif_alt_pressed = False

    def on_smif_pan_press(self, event):
        ax = getattr(self, "ax_smif", None)
        if ax is None or event is None:
            return
        if event.inaxes is not ax:
            return
        if getattr(event, "button", None) != 1:
            return
        if not self._smif_event_has_alt(event):
            return
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return
        try:
            ax.start_pan(float(event_x), float(event_y), 1)
        except Exception:
            return
        self._smif_pan_active = True
        self._smif_pan_button = 1
        self._smif_pan_key = "alt"

    def on_smif_pan_motion(self, event):
        if not bool(getattr(self, "_smif_pan_active", False)):
            return
        ax = getattr(self, "ax_smif", None)
        canvas = getattr(self, "canvas_smif", None)
        if ax is None or canvas is None or event is None:
            return
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return
        try:
            ax.drag_pan(
                int(getattr(self, "_smif_pan_button", 1) or 1),
                str(getattr(self, "_smif_pan_key", "alt") or "alt"),
                float(event_x),
                float(event_y),
            )
        except Exception:
            return
        ax.set_anchor('C')
        self._refresh_smif_main_axis_ticks(ax)
        canvas.draw_idle()

    def on_smif_pan_release(self, event):
        del event
        if not bool(getattr(self, "_smif_pan_active", False)):
            return
        ax = getattr(self, "ax_smif", None)
        if ax is not None:
            try:
                ax.end_pan()
            except Exception:
                pass
            try:
                ax.set_anchor('C')
            except Exception:
                pass
            self._refresh_smif_main_axis_ticks(ax)
        self._smif_pan_active = False
        self._smif_alt_pressed = False

    def _resolve_smif_zoom_anchor(self, ax, event):
        event_x = getattr(event, "x", None)
        event_y = getattr(event, "y", None)
        if event_x is None or event_y is None:
            return None

        cached_points = getattr(self, "_smif_interaction_points", None)
        if isinstance(cached_points, np.ndarray) and cached_points.ndim == 2 and cached_points.shape[1] == 3 and cached_points.size > 0:
            coords = np.asarray(cached_points, dtype=float)
        else:
            points = self._collect_smif_display_points()
            if not points:
                return None
            coords = self._transform_smif_coords(np.asarray(points, dtype=float))

        if coords.ndim != 2 or coords.shape[1] != 3 or coords.size == 0:
            return None

        max_candidates = 12000
        if len(coords) > max_candidates:
            sample_indices = np.linspace(0, len(coords) - 1, num=max_candidates, dtype=int)
            coords = coords[sample_indices]

        try:
            proj_x, proj_y, _proj_z = proj3d.proj_transform(coords[:, 0], coords[:, 1], coords[:, 2], ax.get_proj())
            screen_xy = ax.transData.transform(np.column_stack([proj_x, proj_y]))
        except Exception:
            return None

        deltas = screen_xy - np.asarray([float(event_x), float(event_y)], dtype=float)
        distance_sq = np.einsum("ij,ij->i", deltas, deltas)
        if distance_sq.size == 0:
            return None

        nearest_index = int(np.argmin(distance_sq))
        if not np.isfinite(distance_sq[nearest_index]) or float(distance_sq[nearest_index]) > (160.0 ** 2):
            return None
        anchor = coords[nearest_index]
        return float(anchor[0]), float(anchor[1]), float(anchor[2])

    def on_smif_widget_mousewheel(self, event):
        if not hasattr(self, "ax_smif"):
            return
        if hasattr(event, "delta") and event.delta:
            button = "up" if event.delta > 0 else "down"
        elif hasattr(event, "num") and event.num in (4, 5):
            button = "up" if event.num == 4 else "down"
        else:
            return

        class _Evt:
            pass

        zoom_event = _Evt()
        zoom_event.inaxes = self.ax_smif
        zoom_event.button = button
        zoom_event.x = getattr(event, "x", None)
        zoom_event.y = getattr(event, "y", None)
        self.on_smif_scroll_zoom(zoom_event)
        return "break"

    def _identify_model_parameters_from_measurement_core(self, save_strategy="prompt", refresh_preview=True):
        """基于实验实测 + 工艺信息文件，剔除空载点后按切削样本逐点辨识 K_c / K_e。"""
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            raise ValueError("当前不是实验实测模式，请先导入实验实测文件")
        if not getattr(self, "manual_measurement_data", None):
            raise ValueError("请先导入实验实测文件")
        if not self.get_primary_input_file():
            raise ValueError("请先绑定并处理当前程序的工艺信息文件")
        if not self.data:
            if not self._process_current_input_for_preview():
                raise ValueError("当前工艺信息文件尚未处理成功")

        measurement_case_signature = self._get_current_measurement_case_signature()
        self.measurement_case_signature = measurement_case_signature
        self._clear_imported_profile_state(
            clear_active=bool(self._get_profile_origin() == "imported_profile"),
            reason="reidentify_reset",
        )
        self._set_profile_import_skip_state(skipped=True)
        self._clear_runtime_identified_profile_state(clear_active=True, reason="reidentify_reset")
        if hasattr(self, "kc_coeff"):
            self.kc_coeff.set("")
        if hasattr(self, "ke_coeff"):
            self.ke_coeff.set("")
        if hasattr(self, "kc_sigma"):
            self.kc_sigma.set(0.0)
        self.step_feed_model_signature = ""
        self.profile_origin = "no_profile"
        self.prediction_source = "no_profile"
        self._invalidate_measurement_runtime_state(
            keep_profile_lock=False,
            clear_interval_state=not self._has_authoritative_segmentation_state(),
        )
        self._invalidate_process_alignment_caches(reason="reidentify_reset")
        self._debug_prediction_state_event(
            "reidentify_reset",
            measurement_case_signature=measurement_case_signature or "none",
            interval_policy="recompute_current",
            reverse_solve=True,
            reused_current_template=False,
            kc_map_source="current_rows",
        )
        self._force_recompute_kc_profile = True
        try:
            if hasattr(self, "set_progress"):
                self.set_progress(46, "正在构建样本级实测样本...")
            sample_df = self._build_manual_measurement_sample_frame(
                allow_saved_sample_profile=False,
            )
            if sample_df is None or sample_df.empty:
                raise ValueError("未生成可用于辨识的样本级预测数据")
            sigma_idle, delta_mrr, _idle_count, idle_mask = self._estimate_idle_sigma_and_delta_mrr(
                sample_df,
                kc_reference=self._resolve_measurement_gate_reference_kc(),
            )
            sample_df = self._append_manual_measurement_impedance(sample_df, sigma_idle, delta_mrr, idle_mask)

            if hasattr(self, "set_progress"):
                self.set_progress(50, "正在剔除空载点并按切削样本辨识参数...")
            fit_df = self._build_measurement_identification_fit_frame(sample_df)
            if hasattr(self, "set_progress"):
                self.set_progress(62, f"已选取 {len(fit_df)} 个切削样本点，正在辨识参数...")
            fit_result = self._fit_kc_ke_by_linear_system(
                fit_df["mrr"],
                fit_df["ap"],
                fit_df["cutting_load"],
                "切削样本点",
                fixed_ke=self._resolve_fixed_ke_for_identification(),
            )

            if hasattr(self, "set_progress"):
                self.set_progress(66, "参数辨识完成，正在写回预测负载...")
            idle_commit_result = self._commit_current_program_idle_power()
            self._apply_cutting_model_fit_result(fit_result)
            self._persist_app_config()

            process_path = self.get_primary_input_file()
            process_mtime = 0.0
            measurement_path = getattr(self, "manual_measurement_path", "") or ""
            measurement_mtime = 0.0
            try:
                process_mtime = os.path.getmtime(process_path)
            except Exception:
                pass
            try:
                measurement_mtime = os.path.getmtime(measurement_path)
            except Exception:
                pass
            self.step_feed_model_signature = (
                f"measurement|{os.path.abspath(process_path)}|{process_mtime:.6f}|"
                f"{os.path.abspath(measurement_path) if measurement_path else ''}|{measurement_mtime:.6f}|"
                f"{fit_result['identification_mode']}|"
                f"{fit_result['kc_value']:.6f}|{fit_result['ke_value']:.6f}|{fit_result['kc_sigma']:.6f}"
            )
            self._sync_prediction_mode_after_model_change(prefer_posterior=True)

            ap_values = fit_df["ap"].to_numpy(dtype=float)
            ae_values = fit_df["ae"].to_numpy(dtype=float)
            feed_values = fit_df["feed_speed"].to_numpy(dtype=float)
            specific_values = fit_df["specific_mrr"].to_numpy(dtype=float)
            sigma_idle = float(getattr(self, "manual_measurement_data", {}).get("sigma_idle", 0.0) or 0.0)
            delta_mrr = float(getattr(self, "manual_measurement_data", {}).get("delta_mrr", 0.0) or 0.0)
            idle_point_count = int(np.sum(sample_df["is_idle_point"].to_numpy(dtype=bool))) if "is_idle_point" in sample_df.columns else 0
            self.step_feed_status_var.set(
                f"已按实验实测逐点辨识({self._format_identification_mode_text(fit_result)}): "
                f"切削点={len(fit_df)}, 空载点={idle_point_count}, "
                f"ap={np.min(ap_values):.6f}~{np.max(ap_values):.6f}, "
                f"ae={np.min(ae_values):.6f}~{np.max(ae_values):.6f}, "
                f"F={np.min(feed_values):.3f}~{np.max(feed_values):.3f}, "
                f"ae·F/60={np.min(specific_values):.3f}~{np.max(specific_values):.3f}, "
                f"{self._format_idle_commit_text(idle_commit_result)}, 全局K_c={fit_result['kc_value']:.6f}, "
                f"{self._format_identification_ke_text(fit_result)}, "
                f"σ_Kc={fit_result['kc_sigma']:.6f}, K_c^UCB={fit_result['kc_ucb']:.6f}, "
                f"σ_idle={sigma_idle:.6f}, δ_MRR={delta_mrr:.6f}"
            )
            self.set_status("实验实测模型参数辨识完成", 4000)

            self._refresh_manual_measurement_prediction(
                allow_saved_sample_profile=False,
                allow_measurement_resolve=True,
                display_mode="posterior",
            )
            interval_policy = "recompute_current"
            self.generate_plots(
                save=False,
                silent=True,
                interval_policy=interval_policy,
                persist_profile=False,
                refresh_prediction=False,
            )
            self._persist_current_kc_profile(source="measurement")
            if hasattr(self, "_refresh_current_process_prediction_from_runtime"):
                self._refresh_current_process_prediction_from_runtime(
                    allow_profile_fallback=True,
                    prefer_current_state=False,
                )
            self._debug_prediction_state_event(
                "reidentify_complete",
                measurement_case_signature=measurement_case_signature or "none",
                interval_policy=interval_policy,
                reverse_solve=True,
                display_mode="posterior",
                live_display="posterior",
                reused_current_template=False,
                kc_map_source="runtime_fit",
            )
            if save_strategy == "prompt":
                self._prompt_save_profile_after_identification()
            else:
                self._persist_identified_profile(save_strategy=save_strategy, persist_current=False)
            return fit_result
        finally:
            self._force_recompute_kc_profile = False

    def identify_model_parameters_from_measurement(self):
        return self._identify_model_parameters_from_measurement_core(save_strategy="prompt", refresh_preview=True)

    def identify_model_parameters(self):
        """统一的模型参数辨识入口。实验实测模式优先自动辨识，否则回退到阶梯进给CSV。"""
        if getattr(self, "_auto_identifying_model", False):
            return

        measurement_mode = getattr(self, "sample_data_mode", "") == "experiment_measurement"
        self._arm_model_param_commit_refresh_suppression(duration_seconds=30.0 if measurement_mode else 1.0)
        self._auto_identifying_model = True
        try:
            _, invalid_fields = self._normalize_model_param_inputs_for_runtime()
            self._persist_app_config()
            if invalid_fields and hasattr(self, "set_status"):
                self.set_status(f"{'、'.join(invalid_fields)} 输入无效，已按空值处理", 4000)
            if measurement_mode and hasattr(self, "set_progress"):
                self.set_progress(8, "正在重新辨识模型参数...")
            if measurement_mode:
                if (
                    hasattr(self, "_is_imported_profile_forward_lock_active")
                    and self._is_imported_profile_forward_lock_active()
                ):
                    proceed = messagebox.askyesno(
                        "清除导入配置并重新辨识",
                        "当前已导入参数配置。重新辨识会清除导入配置，并按当前实测负载重新计算 K_c/K_e 和稳态区间。是否继续？"
                    )
                    if not proceed:
                        if hasattr(self, "set_status"):
                            self.set_status("已保持导入参数配置，仅执行前向预测", 4000)
                        return
                self.identify_model_parameters_from_measurement()
                if hasattr(self, "set_progress"):
                    self.set_progress(100, "模型参数重新辨识完成")
            else:
                self.identify_step_feed_parameters()
        except Exception as exc:
            if measurement_mode:
                if hasattr(self, "set_progress"):
                    self.set_progress(0, f"模型参数重新辨识失败: {str(exc)[:60]}")
                self.step_feed_status_var.set(f"自动辨识失败: {str(exc)}")
                messagebox.showerror("辨识失败", f"模型参数辨识失败:\n{str(exc)}")
            else:
                messagebox.showerror("辨识失败", f"模型参数辨识失败:\n{str(exc)}")
        finally:
            self._auto_identifying_model = False
            self._release_model_param_commit_refresh_suppression()
            if measurement_mode and hasattr(self, "reset_progress"):
                try:
                    self.root.after(1200, self.reset_progress)
                except Exception:
                    try:
                        self.reset_progress()
                    except Exception:
                        pass

    def identify_step_feed_parameters(self):
        """导入阶梯进给CSV，辨识K_c与K_e。"""
        file_path = filedialog.askopenfilename(
            title="选择阶梯进给CSV",
            filetypes=(("CSV文件", "*.csv"), ("文本文件", "*.txt"), ("所有文件", "*.*"))
        )
        if not file_path:
            return

        try:
            df = self._read_csv_flex(file_path)
            feed_col = self._find_matching_column(
                df, ["feed_rate", "feed", "进给速度", "进给率", "进给"], fallback_index=0
            )
            power_col = self._find_matching_column(
                df, ["power", "主轴功率", "功率", "load_power"], fallback_index=1
            )
            ap_col = self._find_matching_column(df, ["ap", "a_p", "切深", "轴向切深"])
            ae_col = self._find_matching_column(df, ["ae", "a_e", "切宽", "径向切宽"])
            speed_col = self._find_matching_column(df, ["spindle_speed", "speed", "rpm", "主轴转速", "转速"])

            if not feed_col or not power_col:
                raise ValueError("未识别到进给列或功率列，请检查CSV表头")

            step_df = df[[feed_col, power_col]].copy()
            step_df.columns = ["feed", "power"]
            step_df["feed"] = pd.to_numeric(step_df["feed"], errors='coerce')
            step_df["power"] = pd.to_numeric(step_df["power"], errors='coerce')
            step_df = step_df.dropna()
            if len(step_df) < 2:
                raise ValueError("有效阶梯进给样本不足，至少需要2个数据点")

            ap_val, ae_val = self._resolve_step_feed_geometry(df, ap_col, ae_col)
            if ap_val <= 0 or ae_val <= 0:
                raise ValueError("阶梯进给识别所需的ap/ae必须大于0")

            x = step_df["feed"].to_numpy(dtype=float).reshape(-1, 1)
            y = step_df["power"].to_numpy(dtype=float)

            slope = 0.0
            intercept = 0.0
            try:
                sklearn_mod = _get_sklearn()
                model = sklearn_mod.HuberRegressor()
                model.fit(x, y)
                slope = float(model.coef_[0])
                intercept = float(model.intercept_)
            except Exception:
                slope, intercept = np.polyfit(step_df["feed"].to_numpy(dtype=float), y, 1)
                slope = float(slope)
                intercept = float(intercept)

            y_pred = intercept + slope * x[:, 0]
            residuals = y - y_pred
            sample_count = len(step_df)
            sxx = float(np.sum((x[:, 0] - np.mean(x[:, 0])) ** 2))
            if sample_count > 2 and sxx > 1e-12:
                residual_var = float(np.sum(residuals ** 2) / max(sample_count - 2, 1))
                slope_stderr = math.sqrt(max(residual_var, 0.0) / sxx)
            else:
                slope_stderr = 0.0

            idle_reference = float(self.current_program_idle_power.get() or self.p_idle_var.get())
            if speed_col and self.idle_power_model:
                speed_series = pd.to_numeric(df[speed_col], errors='coerce').dropna()
                if not speed_series.empty:
                    idle_reference = float(np.median([self.predict_idle_power(val) for val in speed_series]))

            fixed_ke = self._resolve_fixed_ke_for_identification()
            if fixed_ke is not None:
                mrr_values = step_df["feed"].to_numpy(dtype=float) * ap_val * ae_val / 60.0
                fit_result = self._fit_kc_ke_by_linear_system(
                    mrr_values,
                    np.full_like(mrr_values, ap_val, dtype=float),
                    y - idle_reference,
                    "阶梯进给CSV",
                    fixed_ke=fixed_ke,
                )
            else:
                kc_value = max(slope * 60.0 / (ap_val * ae_val), 0.0)
                kc_sigma = max(slope_stderr * 60.0 / (ap_val * ae_val), 0.0)
                ke_value = max((intercept - idle_reference) / ap_val, 0.0)
                fit_result = {
                    "kc_value": float(kc_value),
                    "ke_value": float(ke_value),
                    "kc_sigma": float(max(kc_sigma, 0.0)),
                    "kc_ucb": float(kc_value + float(self.kc_beta.get()) * max(kc_sigma, 0.0)),
                    "interval_count": int(sample_count),
                    "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
                    "condition_number": float("nan"),
                    "ke_locked": False,
                    "identification_mode": "kc_ke",
                }

            self.step_feed_csv_path_var.set(file_path)
            idle_commit_result = self._commit_current_program_idle_power()
            self._apply_cutting_model_fit_result(fit_result)
            self._persist_app_config()
            try:
                model_mtime = os.path.getmtime(file_path)
            except Exception:
                model_mtime = 0.0
            self.step_feed_model_signature = (
                f"{os.path.abspath(file_path)}|{model_mtime:.6f}|{fit_result['identification_mode']}|"
                f"{fit_result['kc_value']:.6f}|{fit_result['ke_value']:.6f}|{fit_result['kc_sigma']:.6f}"
            )
            self._sync_prediction_mode_after_model_change(prefer_posterior=True)

            self.step_feed_status_var.set(
                f"参数已辨识({self._format_identification_mode_text(fit_result)}): "
                f"{self._format_idle_commit_text(idle_commit_result)}, 全局K_c={fit_result['kc_value']:.6f}, "
                f"{self._format_identification_ke_text(fit_result)}, "
                f"σ_Kc={fit_result['kc_sigma']:.6f}, K_c^UCB={fit_result['kc_ucb']:.6f}"
            )
            self.set_status("阶梯进给模型参数辨识完成", 3000)

            self._refresh_manual_measurement_prediction(
                allow_saved_sample_profile=False,
                allow_measurement_resolve=True,
            )
            if self.get_primary_input_file():
                self._force_recompute_kc_profile = True
                try:
                    self.generate_plots(
                        save=False,
                        silent=True,
                        interval_policy="reuse_current_template",
                        persist_profile=False,
                    )
                finally:
                    self._force_recompute_kc_profile = False
                self._debug_interval_state_event(
                    "reidentify_overwrite_current_state",
                    source="step_feed",
                    interval_count=len(self._get_current_interval_records(allow_profile_fallback=False)),
                    segment_count=len(self._get_current_segment_records(allow_profile_fallback=False)),
                    profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
                )
            self._persist_current_kc_profile(source="step_feed")
            self._prompt_save_profile_after_identification()
        except Exception as e:
            messagebox.showerror("辨识失败", f"模型参数辨识失败:\n{str(e)}")

    def refresh_pit_button_state(self):
        if not hasattr(self, 'pit_display_btn'):
            return
        has_interval_records = bool(self._get_current_interval_records(allow_profile_fallback=False))
        has_point_data = False
        if not has_interval_records:
            try:
                has_point_data = not self.build_current_pit_dataframe("all").empty
            except Exception:
                has_point_data = bool(getattr(self, "data", None))
        state = "normal" if (has_interval_records or has_point_data) else "disabled"
        self.pit_display_btn.configure(state=state)

    def _format_optional_float(self, value):
        if value is None:
            return ""
        try:
            text = f"{float(value):.6f}"
        except Exception:
            return ""
        text = text.rstrip("0").rstrip(".")
        return text if text else "0"

    def _get_optional_float_value(self, value):
        raw = str(value).strip()
        if not raw:
            return None
        try:
            numeric = float(raw)
        except Exception:
            return None
        return numeric if numeric >= 0 else None

    def get_current_pit_metadata(self):
        return {
            "tool_diameter": self._get_optional_float_value(self.tool_diameter.get()),
            "tool_radius": self._get_optional_float_value(self.tool_radius.get()),
            "tool_material": str(self.workpiece_material.get()).strip(),
            "blank_material": str(self.blank_material.get()).strip(),
        }

    def _sync_pit_metadata_to_records(self):
        if self._is_imported_profile_forward_lock_active():
            return False
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        if not interval_records:
            return False
        metadata = self.get_current_pit_metadata()
        updated_intervals = []
        for record in interval_records:
            if not isinstance(record, dict):
                continue
            current = dict(record)
            current.update(metadata)
            updated_intervals.append(current)
        self._set_current_interval_state(
            interval_records=updated_intervals,
            segment_records=self._get_current_segment_records(allow_profile_fallback=False),
            point_kc_map=dict(getattr(self, "current_interval_point_kc_map", {}) or {}),
            source=str(getattr(self, "_current_interval_source", "") or ""),
            profile_locked=bool(getattr(self, "_profile_intervals_locked", False)),
        )
        return True

    def on_pit_metadata_commit(self, event=None):
        invalid_fields = []
        diameter_value = None
        radius_raw = str(self.tool_radius.get()).strip()
        for label, var in (
            ("刀具直径", self.tool_diameter),
            ("刀具半径", self.tool_radius),
        ):
            raw = str(var.get()).strip()
            if not raw:
                var.set("")
                continue
            value = self._get_optional_float_value(raw)
            if value is None:
                invalid_fields.append(label)
                continue
            var.set(self._format_optional_float(value))
            if label == "刀具直径":
                diameter_value = value

        if diameter_value is None:
            diameter_value = self._get_optional_float_value(self.tool_diameter.get())
        if diameter_value is not None and not radius_raw:
            self.tool_radius.set(self._format_optional_float(diameter_value / 2.0))

        self.workpiece_material.set(str(self.workpiece_material.get()).strip())
        self.blank_material.set(str(self.blank_material.get()).strip())
        self._sync_pit_metadata_to_records()
        self._persist_app_config()
        if invalid_fields:
            self.set_status(f"{'、'.join(invalid_fields)}输入无效，已按空值处理", 4000)

    def show_pit_dialog(self):
        """显示完整PIT表。"""
        point_df = self.build_current_pit_dataframe("all")
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        if point_df.empty and not interval_records:
            messagebox.showwarning("无PIT", "请先导入工艺信息文件")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("完整PIT")
        dialog.geometry("1680x720")
        dialog.minsize(1320, 520)
        dialog.transient(self.root)
        dialog.grab_set()
        center_dialog_on_parent(dialog, self.root)

        summary_columns = [
            ("zone_id", "Zone_ID", 90),
            ("start_label", "StartPt", 100),
            ("end_label", "EndPt", 100),
            ("start_s", "Start_s", 90),
            ("end_s", "End_s", 90),
            ("tool_diameter", "ToolDia", 90),
            ("tool_radius", "ToolR", 90),
            ("tool_material", "ToolMaterial", 150),
            ("blank_material", "BlankMaterial", 150),
            ("a_p", "a_p", 80),
            ("a_e", "a_e", 80),
            ("F_plan", "F_plan", 90),
            ("p_idle", "P_idle", 90),
            ("p_meas", "P_meas", 90),
            ("p_pred", "P_pred", 90),
            ("K_c_hat", "K_c_hat", 100),
            ("sigma_Kc", "sigma_Kc", 100),
            ("K_c_UCB", "K_c_UCB", 100),
            ("sample_count", "SampleCount", 100),
            ("valid_kc_count", "ValidKc", 90),
            ("gated_out_count", "GatedOut", 90),
            ("actual_load_std", "LoadStd", 90),
            ("actual_load_diff_std", "DiffStd", 90),
            ("sigma_idle", "σ_idle", 90),
            ("delta_mrr", "δ_MRR", 90),
            ("kc_source", "KcSource", 120),
        ]

        outer = ttk.Frame(dialog, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=1)

        meta_frame = ttk.LabelFrame(outer, text="PIT元数据", padding=8)
        meta_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        meta_frame.grid_columnconfigure(5, weight=1)
        meta_frame.grid_columnconfigure(7, weight=1)

        ttk.Label(meta_frame, text="刀具直径(mm):", font=UI_FONT_NORMAL).grid(row=0, column=0, sticky="w", padx=(0, 4))
        tool_diameter_entry = ttk.Entry(meta_frame, textvariable=self.tool_diameter, width=12, font=UI_FONT_NORMAL)
        tool_diameter_entry.grid(row=0, column=1, sticky="w", padx=(0, 10))

        ttk.Label(meta_frame, text="刀具半径(mm):", font=UI_FONT_NORMAL).grid(row=0, column=2, sticky="w", padx=(0, 4))
        tool_radius_entry = ttk.Entry(meta_frame, textvariable=self.tool_radius, width=12, font=UI_FONT_NORMAL)
        tool_radius_entry.grid(row=0, column=3, sticky="w", padx=(0, 10))

        ttk.Label(meta_frame, text="刀具材料:", font=UI_FONT_NORMAL).grid(row=0, column=4, sticky="w", padx=(0, 4))
        tool_material_entry = ttk.Entry(meta_frame, textvariable=self.workpiece_material, width=18, font=UI_FONT_NORMAL)
        tool_material_entry.grid(row=0, column=5, sticky="ew", padx=(0, 10))

        ttk.Label(meta_frame, text="毛坯材料:", font=UI_FONT_NORMAL).grid(row=0, column=6, sticky="w", padx=(0, 4))
        blank_material_entry = ttk.Entry(meta_frame, textvariable=self.blank_material, width=18, font=UI_FONT_NORMAL)
        blank_material_entry.grid(row=0, column=7, sticky="ew", padx=(0, 10))

        meta_hint_var = tk.StringVar(value="可手动填写或修改；导入NC时自动读取显式直径字段，并按直径一半回填刀具半径。")
        ttk.Label(
            meta_frame,
            textvariable=meta_hint_var,
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED
        ).grid(row=1, column=0, columnspan=9, sticky="w", pady=(6, 0))

        notebook = ttk.Notebook(outer)
        notebook.grid(row=1, column=0, sticky="nsew")

        point_tab = ttk.Frame(notebook, padding=8)
        summary_tab = ttk.Frame(notebook, padding=8)
        notebook.add(point_tab, text="工艺点明细")
        notebook.add(summary_tab, text="区间汇总")

        def _build_tree(parent):
            container = ttk.Frame(parent)
            container.pack(fill=tk.BOTH, expand=True)
            tree = ttk.Treeview(container, show="headings")
            vsb = ttk.Scrollbar(container, orient=tk.VERTICAL, command=tree.yview)
            hsb = ttk.Scrollbar(container, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
            container.grid_rowconfigure(0, weight=1)
            container.grid_columnconfigure(0, weight=1)
            return tree

        ttk.Label(
            point_tab,
            text=(
                "累计行程 s 显示统一累计坐标；普通 s(mm) 按逐行增量累加，"
                "只有显式 cumulative 列才按输入累计值解释。"
            ),
            font=UI_FONT_SMALL,
            foreground=UI_COLOR_TEXT_MUTED,
        ).pack(anchor="w", pady=(0, 6))
        point_tree = _build_tree(point_tab)
        summary_tree = _build_tree(summary_tab)

        point_empty_var = tk.StringVar(value="")
        summary_empty_var = tk.StringVar(value="")
        ttk.Label(point_tab, textvariable=point_empty_var, font=UI_FONT_SMALL, foreground=UI_COLOR_TEXT_MUTED).pack(anchor="w", pady=(6, 0))
        ttk.Label(summary_tab, textvariable=summary_empty_var, font=UI_FONT_SMALL, foreground=UI_COLOR_TEXT_MUTED).pack(anchor="w", pady=(6, 0))

        def _reload_tables():
            current_point_df = self.build_current_pit_dataframe("all")
            point_columns = self._get_pit_display_columns(current_point_df)
            point_view_df = current_point_df.loc[:, point_columns].copy() if point_columns and not current_point_df.empty else current_point_df.copy()
            self._populate_treeview(point_tree, point_view_df)
            point_empty_var.set("" if not point_view_df.empty else "当前没有可显示的点级 PIT 数据。")

            current_intervals = self._get_current_interval_records(allow_profile_fallback=False)
            summary_df = pd.DataFrame(current_intervals)
            if not summary_df.empty:
                for key, _, _ in summary_columns:
                    if key not in summary_df.columns:
                        summary_df[key] = ""
                summary_view_df = summary_df.loc[:, [key for key, _, _ in summary_columns]].copy()
                summary_view_df.columns = [heading for _, heading, _ in summary_columns]
            else:
                summary_view_df = pd.DataFrame(columns=[heading for _, heading, _ in summary_columns])
            self._populate_treeview(summary_tree, summary_view_df)
            if getattr(summary_tree, "_codex_columns", None) == list(summary_view_df.columns):
                for _, heading, width in summary_columns:
                    if heading not in summary_view_df.columns:
                        continue
                    anchor = "w" if heading in {"ToolMaterial", "BlankMaterial", "KcSource"} else "center"
                    summary_tree.column(heading, width=width, anchor=anchor, stretch=True)
            summary_empty_var.set("" if not summary_view_df.empty else "当前还没有稳态区间汇总结果。")

        def _commit_and_refresh(event=None):
            self.on_pit_metadata_commit(event)
            _reload_tables()
            self.refresh_smif_view()
            return None

        ttk.Button(meta_frame, text="应用到当前PIT", command=_commit_and_refresh, width=14).grid(row=0, column=8, sticky="e")

        for widget in (
            tool_diameter_entry,
            tool_radius_entry,
            tool_material_entry,
            blank_material_entry,
        ):
            widget.bind("<Return>", _commit_and_refresh)
            widget.bind("<FocusOut>", _commit_and_refresh)

        _reload_tables()

        ttk.Button(outer, text="关闭", command=dialog.destroy, width=10).grid(row=2, column=0, pady=(8, 0))

    def on_model_param_commit(self, event=None):
        """手动修改P_idle/K_c/K_e后刷新当前预览。"""
        _, invalid_fields = self._normalize_model_param_inputs_for_runtime()
        if self._is_model_param_commit_refresh_suppressed() or getattr(self, "_auto_identifying_model", False):
            self._persist_app_config()
            if invalid_fields:
                self.set_status(f"{'、'.join(invalid_fields)} 输入无效，已按空值处理", 4000)
            return
        if self.has_identified_kc_ke():
            self.step_feed_model_signature = ""
            if hasattr(self, "prediction_mode_var"):
                self.prediction_mode_var.set("direct_prediction")
            self.step_feed_status_var.set(
                f"当前模型参数: K_c={self.get_kc_value():.6f}, K_e={self.get_ke_value():.6f}"
            )
            self._sync_prediction_mode_after_model_change(prefer_posterior=False)
        else:
            self.clear_kc_ke_state(persist=False, status_text="未辨识模型参数")
        self._persist_app_config()
        self._refresh_manual_measurement_prediction()

        if self.get_primary_input_file():
            self._process_current_input_for_preview()
        elif self.data:
            interval_policy = self._get_default_interval_policy()
            self.generate_plots(save=False, silent=True, interval_policy=interval_policy)
        if invalid_fields:
            self.set_status(f"{'、'.join(invalid_fields)} 输入无效，已按空值处理", 4000)
