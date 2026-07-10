from __future__ import annotations

from .shared import *


class PlotSupportMixin:
    def _compress_plot_segment_preserve_extrema(self, x_segment, y_segment, max_points):
        """按桶保留边界/峰值/谷值，压缩顶点数但尽量保留完整波形信息。"""
        try:
            limit = int(max_points)
        except Exception:
            limit = 0
        if limit <= 0:
            return x_segment, y_segment

        x_arr = np.asarray(x_segment)
        y_arr = np.asarray(y_segment)
        point_count = len(x_arr)
        if point_count <= limit or point_count <= 4:
            return x_arr, y_arr

        bucket_count = max(1, min(point_count, limit // 4))
        if bucket_count <= 1:
            return x_arr[[0, -1]], y_arr[[0, -1]]

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
            local_y = y_arr[start:end]
            finite_local = np.flatnonzero(np.isfinite(local_y))
            if finite_local.size > 0:
                finite_values = local_y[finite_local]
                local_min = int(finite_local[int(np.argmin(finite_values))]) + start
                local_max = int(finite_local[int(np.argmax(finite_values))]) + start
                local_indices.add(local_min)
                local_indices.add(local_max)

            for idx in sorted(local_indices):
                _append(idx)

        _append(point_count - 1)
        if len(kept_indices) >= point_count:
            return x_arr, y_arr
        return x_arr[kept_indices], y_arr[kept_indices]

    def compute_contiguous_blocks(self, mask):
        """将布尔掩码转换为连续区块索引列表 [(start, end), ...]。"""
        if mask is None:
            return []
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.size == 0:
            return []

        blocks = []
        start = None
        for idx, flag in enumerate(mask_arr):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                blocks.append((start, idx - 1))
                start = None
        if start is not None:
            blocks.append((start, len(mask_arr) - 1))
        return blocks

    def compute_sequence_blocks(self, line_numbers, group_keys=None):
        """按时间顺序切分序列块。

        满足以下任一条件时开启新区块：
        1. 分组键变化（如程序号切换）
        2. 行号下降（典型的下一次循环/下一段开始）
        """
        if line_numbers is None:
            return []

        line_arr = np.asarray(line_numbers)
        if line_arr.size == 0:
            return []

        group_arr = np.asarray(group_keys) if group_keys is not None else None
        blocks = []
        start = 0
        for idx in range(1, len(line_arr)):
            split = False
            if group_arr is not None and group_arr[idx] != group_arr[idx - 1]:
                split = True
            else:
                try:
                    split = int(line_arr[idx]) < int(line_arr[idx - 1])
                except Exception:
                    split = line_arr[idx] != line_arr[idx - 1]
            if split:
                blocks.append((start, idx - 1))
                start = idx
        blocks.append((start, len(line_arr) - 1))
        return blocks

    def _compute_line_x_positions_single_block(self, line_numbers):
        """基于行号生成可视化x坐标
        
        当多个数据点具有相同的行号时，将它们按锚点语义平均排布在单位区间内。
        例如：行号11有3个点，则分布在 11.0, 11.333, 11.667。
        """
        if line_numbers is None:
            return []
        line_point_counts = {}
        point_indices = []
        for ln in line_numbers:
            count = line_point_counts.get(ln, 0) + 1
            line_point_counts[ln] = count
            point_indices.append(count)
        x_positions = []
        for ln, idx in zip(line_numbers, point_indices):
            total_pts = line_point_counts.get(ln, 1)
            if total_pts <= 1:
                # 只有一个点，放在行号位置
                x_positions.append(float(ln))
            else:
                # 多个点均匀分布在 [ln, ln+1) 区间内
                # 第1个点在 ln，最后一个点在 ln + (total_pts-1)/total_pts
                x_positions.append(float(ln) + (idx - 1) / total_pts)
        return x_positions

    def compute_line_x_positions(self, line_numbers, blocks=None):
        """基于行号生成可视化x坐标，可按区块独立重置。"""
        if line_numbers is None:
            return []
        line_list = list(line_numbers)
        if not line_list:
            return []
        if not blocks:
            return self._compute_line_x_positions_single_block(line_list)

        x_positions = [math.nan] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_x = self._compute_line_x_positions_single_block(line_list[start:end + 1])
            x_positions[start:end + 1] = local_x
        return x_positions

    def _compute_line_segment_bounds_single_block(self, line_numbers):
        """计算每行工艺信息对应的区间边界 [start_x, end_x]
        
        每行占据一段区间而非一个点：
        - 行号11只有1行 → [11, 12)
        - 行号11有3行 → [11, 11.333), [11.333, 11.667), [11.667, 12)
        
        返回: (start_bounds, end_bounds) 两个列表
        """
        if not line_numbers:
            return [], []
        
        # 统计每个行号有多少行
        from collections import Counter
        line_counts = Counter(line_numbers)
        line_indices = {}
        
        start_bounds = []
        end_bounds = []
        
        for ln in line_numbers:
            idx = line_indices.get(ln, 0)
            total = line_counts[ln]
            
            start_x = float(ln) + idx / total
            end_x = float(ln) + (idx + 1) / total
            
            start_bounds.append(start_x)
            end_bounds.append(end_x)
            
            line_indices[ln] = idx + 1
        
        return start_bounds, end_bounds

    def compute_line_segment_bounds(self, line_numbers, blocks=None):
        """计算每行工艺信息对应的区间边界 [start_x, end_x]，支持按区块独立重置。"""
        if line_numbers is None:
            return [], []
        line_list = list(line_numbers)
        if not line_list:
            return [], []
        if not blocks:
            return self._compute_line_segment_bounds_single_block(line_list)

        start_bounds = [math.nan] * len(line_list)
        end_bounds = [math.nan] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_start, local_end = self._compute_line_segment_bounds_single_block(line_list[start:end + 1])
            start_bounds[start:end + 1] = local_start
            end_bounds[start:end + 1] = local_end
        return start_bounds, end_bounds

    def compute_line_point_widths(self, line_numbers):
        """计算每个点在其行号下的宽度
        
        行号L有n个点时，每个点的宽度是1/n
        """
        if line_numbers is None:
            return []
        line_point_counts = {}
        for ln in line_numbers:
            line_point_counts[ln] = line_point_counts.get(ln, 0) + 1
        widths = []
        for ln in line_numbers:
            total_pts = line_point_counts.get(ln, 1)
            widths.append(1.0 / total_pts if total_pts > 0 else 1.0)
        return widths

    def _compute_line_point_indices_single_block(self, line_numbers):
        """计算每个点在其行号下的序号(从0开始)"""
        if line_numbers is None:
            return []
        line_point_counts = {}
        point_indices = []
        for ln in line_numbers:
            count = line_point_counts.get(ln, 0)
            point_indices.append(count)
            line_point_counts[ln] = count + 1
        return point_indices

    def compute_line_point_indices(self, line_numbers, blocks=None):
        """计算每个点在其行号下的序号(从0开始)，支持按区块独立重置。"""
        if line_numbers is None:
            return []
        line_list = list(line_numbers)
        if not line_list:
            return []
        if not blocks:
            return self._compute_line_point_indices_single_block(line_list)

        point_indices = [0] * len(line_list)
        for start, end in blocks:
            if start > end:
                continue
            local_indices = self._compute_line_point_indices_single_block(line_list[start:end + 1])
            point_indices[start:end + 1] = local_indices
        return point_indices

    def plot_series_by_blocks(self, ax, x_values, y_values, blocks, **plot_kwargs):
        """按连续区块分别绘制折线，避免跨块连线。"""
        if x_values is None or y_values is None or not blocks:
            return []

        max_points = plot_kwargs.pop("max_points", None)
        x_arr = np.asarray(x_values)
        y_arr = np.asarray(y_values)
        artists = []
        first = True
        total_points = sum(max(0, end - start + 1) for start, end in blocks if end >= start)
        use_compression = False
        try:
            max_points_int = int(max_points) if max_points is not None else 0
        except Exception:
            max_points_int = 0
        if max_points_int > 0 and total_points > max_points_int:
            use_compression = True
        for start, end in blocks:
            if start > end:
                continue
            block_kwargs = dict(plot_kwargs)
            if not first:
                block_kwargs.pop("label", None)
            block_x = x_arr[start:end + 1]
            block_y = y_arr[start:end + 1]
            if use_compression and len(block_x) > 4:
                block_budget = max(8, int(round(len(block_x) / float(total_points) * float(max_points_int))))
                block_budget = min(len(block_x), block_budget)
                block_x, block_y = self._compress_plot_segment_preserve_extrema(block_x, block_y, block_budget)
            artist = ax.plot(block_x, block_y, **block_kwargs)
            artists.extend(artist)
            first = False
        return artists

    def get_sample_time_indices_array(self):
        """返回 SampleData 的时间轴数组（每点1ms）。"""
        if self.sample_data_line_numbers is None:
            return None
        if self.sample_data_time_indices is None or len(self.sample_data_time_indices) != len(self.sample_data_line_numbers):
            self.sample_data_time_indices = np.arange(len(self.sample_data_line_numbers), dtype=int)
        return np.asarray(self.sample_data_time_indices, dtype=float)

    def build_sample_line_time_spans(self, mask):
        """基于 SampleData 时间轴构建各程序行号的时间跨度。"""
        if self.sample_data_line_numbers is None:
            return {}, []
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return {}, []

        line_arr = np.asarray(self.sample_data_line_numbers, dtype=int)
        mask_arr = np.asarray(mask, dtype=bool) if mask is not None else np.ones(len(line_arr), dtype=bool)
        if mask_arr.size != len(line_arr):
            return {}, []

        spans_by_line = collections.defaultdict(list)
        ordered_spans = []
        for block_start, block_end in self.compute_contiguous_blocks(mask_arr):
            run_start = block_start
            for idx in range(block_start + 1, block_end + 1):
                if line_arr[idx] != line_arr[idx - 1]:
                    ln = int(line_arr[run_start])
                    start_t = float(time_arr[run_start])
                    end_t = float(time_arr[idx - 1] + 1.0)
                    spans_by_line[ln].append((start_t, end_t))
                    ordered_spans.append((ln, start_t, end_t))
                    run_start = idx
            ln = int(line_arr[run_start])
            start_t = float(time_arr[run_start])
            end_t = float(time_arr[block_end] + 1.0)
            spans_by_line[ln].append((start_t, end_t))
            ordered_spans.append((ln, start_t, end_t))
        return spans_by_line, ordered_spans

    def allocate_items_to_spans(self, item_count, spans):
        """按跨度长度将若干项目分配到多个时间跨度。"""
        if item_count <= 0 or not spans:
            return []
        if len(spans) == 1:
            return [(spans[0], item_count)]

        lengths = np.asarray([max(1e-9, end - start) for start, end in spans], dtype=float)
        total = float(np.sum(lengths))
        if total <= 0:
            return [(spans[0], item_count)]

        raw = lengths / total * float(item_count)
        counts = np.floor(raw).astype(int)
        remainders = raw - counts
        assigned = int(np.sum(counts))
        remaining = item_count - assigned

        if remaining > 0:
            order = np.argsort(-remainders)
            for idx in order[:remaining]:
                counts[int(idx)] += 1

        if np.sum(counts) == 0:
            counts[0] = item_count

        allocations = []
        for idx, count in enumerate(counts):
            if count <= 0:
                continue
            allocations.append((spans[idx], int(count)))

        total_allocated = sum(count for _, count in allocations)
        if total_allocated < item_count and allocations:
            span, count = allocations[-1]
            allocations[-1] = (span, count + (item_count - total_allocated))
        return allocations

    def compute_process_time_bounds_from_sample(self, process_line_numbers, sample_mask):
        """将工艺信息行映射到 SampleData 的时间轴跨度。"""
        if process_line_numbers is None:
            return [], []

        process_lines = np.asarray(process_line_numbers, dtype=int)
        start_bounds = [math.nan] * len(process_lines)
        end_bounds = [math.nan] * len(process_lines)
        if process_lines.size == 0:
            return start_bounds, end_bounds

        spans_by_line, _ = self.build_sample_line_time_spans(sample_mask)
        if not spans_by_line:
            return start_bounds, end_bounds

        positions_by_line = collections.OrderedDict()
        for idx, ln in enumerate(process_lines):
            positions_by_line.setdefault(int(ln), []).append(idx)

        for ln, positions in positions_by_line.items():
            spans = spans_by_line.get(int(ln))
            if not spans:
                continue

            allocations = self.allocate_items_to_spans(len(positions), spans)
            pos_cursor = 0
            for span, count in allocations:
                start_t, end_t = span
                width = (end_t - start_t) / count if count > 0 else 0.0
                for local_idx in range(count):
                    if pos_cursor >= len(positions):
                        break
                    proc_idx = positions[pos_cursor]
                    start_bounds[proc_idx] = start_t + local_idx * width
                    end_bounds[proc_idx] = start_t + (local_idx + 1) * width
                    pos_cursor += 1

        return start_bounds, end_bounds

    def build_time_spans_for_ranges(self, ranges, sample_mask):
        """将刀具/程序行号范围转换为 SampleData 时间轴区间。"""
        if not ranges or self.sample_data_line_numbers is None:
            return []
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return []

        line_arr = np.asarray(self.sample_data_line_numbers, dtype=int)
        base_mask = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else np.ones(len(line_arr), dtype=bool)
        if base_mask.size != len(line_arr):
            return []

        range_mask = np.zeros(len(line_arr), dtype=bool)
        for start_line, end_line in ranges:
            range_mask |= (line_arr >= start_line) & (line_arr <= end_line)
        target_mask = base_mask & range_mask

        spans = []
        for block_start, block_end in self.compute_contiguous_blocks(target_mask):
            spans.append((float(time_arr[block_start]), float(time_arr[block_end] + 1.0)))
        return spans

    def remove_line_axis_on_time(self, ax):
        """移除指定坐标轴顶部的程序行号辅助轴。"""
        top_ax = getattr(ax, '_time_line_top_axis', None)
        if top_ax is None:
            return
        try:
            top_ax.remove()
        except Exception:
            pass
        ax._time_line_top_axis = None

    def _get_time_line_view_key(self, ax):
        """返回当前顶部程序行号轴的视图签名，用于检测缩放/平移/重绘后的同步需求。"""
        if ax is None:
            return None
        try:
            x_min, x_max = ax.get_xlim()
            bbox = ax.get_window_extent()
            width_px = int(round(float(getattr(bbox, "width", 0.0))))
        except Exception:
            return None
        if not (np.isfinite(x_min) and np.isfinite(x_max)):
            return None
        axis_mode = str(getattr(ax, "_time_line_axis_mode", "time") or "time")
        return (
            axis_mode,
            round(float(x_min), 6),
            round(float(x_max), 6),
            max(width_px, 0),
        )

    def get_time_line_axis_capacity(self, ax, min_ticks=8, max_ticks=60):
        """根据当前轴宽度估算顶部程序行号可容纳的刻度数。"""
        try:
            bbox = ax.get_window_extent()
            width_px = float(getattr(bbox, 'width', 0.0))
        except Exception:
            width_px = 0.0
        if width_px <= 0:
            return min_ticks
        tick_count = int(width_px // 75)
        return max(min_ticks, min(max_ticks, tick_count))

    def build_line_anchor_spans(self, spans):
        """按唯一程序行号归并跨度，并使用全局中心作为稳定锚点。"""
        aggregated = collections.OrderedDict()
        for span in spans or []:
            if not span or len(span) < 3:
                continue
            try:
                line_no = int(span[0])
                start_value = float(span[1])
                end_value = float(span[2])
            except Exception:
                continue
            if not (np.isfinite(start_value) and np.isfinite(end_value)):
                continue
            span_start = float(min(start_value, end_value))
            span_end = float(max(start_value, end_value))
            current = aggregated.get(line_no)
            if current is None:
                aggregated[line_no] = [span_start, span_end]
            else:
                current[0] = min(float(current[0]), span_start)
                current[1] = max(float(current[1]), span_end)

        anchor_spans = []
        for line_no, (span_start, span_end) in aggregated.items():
            anchor_spans.append((
                int(line_no),
                0.5 * (float(span_start) + float(span_end)),
                float(span_start),
                float(span_end),
            ))
        return anchor_spans

    def sample_time_line_spans(self, spans, target_count):
        """对唯一行号锚点做均匀抽样，优先保留首尾和视窗中心附近。"""
        anchor_spans = self.build_line_anchor_spans(spans)
        if not anchor_spans or target_count <= 0 or len(anchor_spans) <= target_count:
            return anchor_spans

        if target_count == 1:
            center_idx = len(anchor_spans) // 2
            return [anchor_spans[center_idx]]

        selected_indices = {0, len(anchor_spans) - 1}
        center_idx = min(
            range(len(anchor_spans)),
            key=lambda idx: abs(float(anchor_spans[idx][1]) - float(np.mean([anchor_spans[0][1], anchor_spans[-1][1]]))),
        )
        selected_indices.add(int(center_idx))

        if len(selected_indices) < int(target_count):
            extra_indices = np.linspace(0, len(anchor_spans) - 1, num=int(target_count), dtype=int).tolist()
            selected_indices.update(int(idx) for idx in extra_indices)

        ordered_indices = sorted(selected_indices)
        while len(ordered_indices) > int(target_count):
            removable_positions = [pos for pos, idx in enumerate(ordered_indices) if idx not in {0, len(anchor_spans) - 1, center_idx}]
            if not removable_positions:
                removable_positions = list(range(1, max(1, len(ordered_indices) - 1)))
            if not removable_positions:
                break
            remove_pos = removable_positions[-1]
            ordered_indices.pop(remove_pos)

        return [anchor_spans[idx] for idx in ordered_indices]

    def select_time_line_spans(self, spans, target_count):
        """优先保证可见程序行号不被无故抽样掉，再按容量压缩标签。"""
        ordered_spans = self.build_line_anchor_spans(spans)
        try:
            target = int(target_count)
        except Exception:
            target = 0
        if target <= 0 or len(ordered_spans) <= target:
            return ordered_spans
        return self.sample_time_line_spans(ordered_spans, target)

    def on_time_line_axis_xlim_changed(self, ax):
        """横轴变化后刷新顶部程序行号轴。"""
        if getattr(ax, '_time_line_refreshing', False):
            return
        mask = getattr(ax, '_time_line_mask', None)
        if mask is None:
            return
        ax._time_line_refreshing = True
        try:
            axis_mode = getattr(ax, '_time_line_axis_mode', 'time')
            if axis_mode == 'path':
                positions = getattr(ax, '_time_line_positions', None)
                self.apply_line_axis_on_path(ax, positions, mask)
            else:
                self.apply_line_axis_on_time(ax, mask)
        finally:
            ax._time_line_refreshing = False

    def _refresh_time_line_axis_on_draw(self, ax, event=None):
        """兜底处理未触发 xlim 回调的平移/缩放/重绘场景。"""
        if ax is None or getattr(ax, "_time_line_refreshing", False):
            return
        canvas = getattr(getattr(ax, "figure", None), "canvas", None)
        event_canvas = getattr(event, "canvas", None)
        if canvas is not None and event_canvas is not None and canvas is not event_canvas:
            return
        current_key = self._get_time_line_view_key(ax)
        if current_key is None:
            return
        if current_key == getattr(ax, "_time_line_view_key", None):
            return
        self.on_time_line_axis_xlim_changed(ax)

    def bind_time_line_axis_updates(self, ax):
        """确保横轴缩放/平移时自动刷新顶部程序行号轴。"""
        if getattr(ax, "_time_line_bound_figure", None) is not getattr(ax, "figure", None):
            ax._time_line_callback_bound = False
            ax._time_line_draw_event_bound = False
            ax._time_line_link_proxy_bound = False
            ax._time_line_bound_figure = getattr(ax, "figure", None)
        if getattr(ax, '_time_line_callback_bound', False):
            linked_axes = list(getattr(ax, "_time_line_linked_axes", []) or [])
        else:
            linked_axes = list(getattr(ax, "_time_line_linked_axes", []) or [])
            try:
                ax.callbacks.connect('xlim_changed', self.on_time_line_axis_xlim_changed)
                ax._time_line_callback_bound = True
            except Exception:
                pass
            canvas = getattr(getattr(ax, "figure", None), "canvas", None)
            if canvas is not None and not getattr(ax, "_time_line_draw_event_bound", False):
                try:
                    canvas.mpl_connect(
                        "draw_event",
                        lambda event, master_ax=ax: self._refresh_time_line_axis_on_draw(master_ax, event),
                    )
                    ax._time_line_draw_event_bound = True
                except Exception:
                    pass
        for linked_ax in linked_axes:
            if linked_ax is None or linked_ax is ax:
                continue
            if getattr(linked_ax, "_time_line_link_master", None) is not ax:
                linked_ax._time_line_link_proxy_bound = False
                linked_ax._time_line_link_master = ax
            if getattr(linked_ax, "_time_line_link_proxy_bound", False):
                continue
            try:
                linked_ax.callbacks.connect(
                    'xlim_changed',
                    lambda _current_ax, master_ax=ax: self.on_time_line_axis_xlim_changed(master_ax)
                )
                linked_ax._time_line_link_proxy_bound = True
                linked_ax._time_line_link_master = ax
            except Exception:
                continue

    def apply_line_axis_on_time(self, ax, sample_mask, max_ticks=60):
        """在时间轴上方附加程序行号辅助坐标轴。"""
        if self.sample_data_line_numbers is None:
            return None
        time_arr = self.get_sample_time_indices_array()
        if time_arr is None:
            return None

        self.remove_line_axis_on_time(ax)
        mask_arr = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else np.ones(len(self.sample_data_line_numbers), dtype=bool)
        if mask_arr.size != len(self.sample_data_line_numbers):
            return None

        _, ordered_spans = self.build_sample_line_time_spans(mask_arr)
        if not ordered_spans:
            return None

        anchor_spans = self.build_line_anchor_spans(ordered_spans)
        if not anchor_spans:
            return None

        x_min, x_max = ax.get_xlim()
        visible_spans = [
            (ln, anchor_t, start_t, end_t)
            for ln, anchor_t, start_t, end_t in anchor_spans
            if x_min <= float(anchor_t) <= x_max
        ]
        if not visible_spans:
            visible_spans = [
                (ln, anchor_t, start_t, end_t)
                for ln, anchor_t, start_t, end_t in anchor_spans
                if end_t >= x_min and start_t <= x_max
            ]
        if not visible_spans:
            visible_spans = anchor_spans

        adaptive_tick_count = self.get_time_line_axis_capacity(ax, min_ticks=8, max_ticks=max_ticks)
        visible_spans = self.select_time_line_spans(visible_spans, adaptive_tick_count)

        tick_positions = []
        tick_labels = []
        for ln, anchor_t, _start_t, _end_t in visible_spans:
            tick_positions.append(float(anchor_t))
            tick_labels.append(str(int(ln)))

        top_ax = ax.secondary_xaxis('top')
        top_ax.set_xticks(tick_positions)
        top_ax.set_xticklabels(tick_labels, rotation=45, ha='left')
        top_ax.set_xlabel('程序行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color='black')
        top_ax.tick_params(axis='x', labelsize=PLOT_FONT_BASE - 1, colors='black')
        try:
            top_ax.spines['top'].set_color('black')
        except Exception:
            pass
        ax._time_line_mask = mask_arr.copy()
        ax._time_line_axis_mode = 'time'
        ax._time_line_positions = None
        ax._time_line_top_axis = top_ax
        ax._time_line_view_key = self._get_time_line_view_key(ax)
        self.bind_time_line_axis_updates(ax)
        return top_ax

    def build_sample_line_position_spans(self, positions, mask):
        """基于任意位置轴构建各程序行号的跨度。"""
        if self.sample_data_line_numbers is None or positions is None:
            return {}, []

        line_arr = np.asarray(self.sample_data_line_numbers, dtype=int)
        pos_arr = np.asarray(positions, dtype=float)
        if pos_arr.size != len(line_arr):
            return {}, []

        mask_arr = np.asarray(mask, dtype=bool) if mask is not None else np.ones(len(line_arr), dtype=bool)
        if mask_arr.size != len(line_arr):
            return {}, []

        spans_by_line = collections.defaultdict(list)
        ordered_spans = []
        for block_start, block_end in self.compute_contiguous_blocks(mask_arr):
            run_start = block_start
            for idx in range(block_start + 1, block_end + 1):
                prev_pos = float(pos_arr[idx - 1])
                curr_pos = float(pos_arr[idx])
                if line_arr[idx] != line_arr[idx - 1] or not np.isfinite(prev_pos) or not np.isfinite(curr_pos):
                    ln = int(line_arr[run_start])
                    start_pos = float(pos_arr[run_start])
                    end_pos = float(pos_arr[idx - 1])
                    if np.isfinite(start_pos) and np.isfinite(end_pos):
                        if end_pos <= start_pos:
                            end_pos = start_pos
                        spans_by_line[ln].append((start_pos, end_pos))
                        ordered_spans.append((ln, start_pos, end_pos))
                    run_start = idx
            ln = int(line_arr[run_start])
            start_pos = float(pos_arr[run_start])
            end_pos = float(pos_arr[block_end])
            if np.isfinite(start_pos) and np.isfinite(end_pos):
                if end_pos <= start_pos and block_end > run_start:
                    end_pos = float(np.nanmax(pos_arr[run_start:block_end + 1]))
                if end_pos <= start_pos:
                    end_pos = start_pos
                spans_by_line[ln].append((start_pos, end_pos))
                ordered_spans.append((ln, start_pos, end_pos))
        return spans_by_line, ordered_spans

    def apply_line_axis_on_path(self, ax, sample_positions, sample_mask, max_ticks=60):
        """在行程轴上方附加程序行号辅助坐标轴。"""
        if self.sample_data_line_numbers is None or sample_positions is None:
            return None

        self.remove_line_axis_on_time(ax)
        _, ordered_spans = self.build_sample_line_position_spans(sample_positions, sample_mask)
        if not ordered_spans:
            return None

        anchor_spans = self.build_line_anchor_spans(ordered_spans)
        if not anchor_spans:
            return None

        x_min, x_max = ax.get_xlim()
        visible_spans = [
            (ln, anchor_pos, start_pos, end_pos)
            for ln, anchor_pos, start_pos, end_pos in anchor_spans
            if x_min <= float(anchor_pos) <= x_max
        ]
        if not visible_spans:
            visible_spans = [
                (ln, anchor_pos, start_pos, end_pos)
                for ln, anchor_pos, start_pos, end_pos in anchor_spans
                if end_pos >= x_min and start_pos <= x_max
            ]
        if not visible_spans:
            visible_spans = anchor_spans

        adaptive_tick_count = self.get_time_line_axis_capacity(ax, min_ticks=8, max_ticks=max_ticks)
        visible_spans = self.select_time_line_spans(visible_spans, adaptive_tick_count)

        tick_positions = []
        tick_labels = []
        for ln, anchor_pos, _start_pos, _end_pos in visible_spans:
            tick_positions.append(float(anchor_pos))
            tick_labels.append(str(int(ln)))

        top_ax = ax.secondary_xaxis('top')
        top_ax.set_xticks(tick_positions)
        top_ax.set_xticklabels(tick_labels, rotation=45, ha='left')
        top_ax.set_xlabel('程序行号', fontsize=PLOT_FONT_BASE, fontweight='bold', color='black')
        top_ax.tick_params(axis='x', labelsize=PLOT_FONT_BASE - 1, colors='black')
        try:
            top_ax.spines['top'].set_color('black')
        except Exception:
            pass
        ax._time_line_mask = np.asarray(sample_mask, dtype=bool) if sample_mask is not None else None
        ax._time_line_axis_mode = 'path'
        ax._time_line_positions = np.asarray(sample_positions, dtype=float) if sample_positions is not None else None
        ax._time_line_top_axis = top_ax
        ax._time_line_view_key = self._get_time_line_view_key(ax)
        self.bind_time_line_axis_updates(ax)
        return top_ax

    def apply_line_axis(self, ax, line_numbers):
        """设置对齐行号刻度 - 使用自动定位器实现动态刻度调整"""
        if line_numbers is None or len(line_numbers) == 0:
            return
        unique_lines = sorted(set(line_numbers))
        if not unique_lines:
            return
        
        # 使用 MaxNLocator 实现自动刻度调整，避免刻度过密或过疏
        # nbins='auto' 会根据可用空间自动调整刻度数量
        # integer=True 确保刻度值为整数
        ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', integer=True, prune='both'))
        
        # 设置刻度标签样式
        ax.tick_params(axis='x', rotation=45, labelsize=PLOT_FONT_BASE, colors=PLOT_TEXT_COLOR)
        
        # 调整刻度标签对齐方式
        for label in ax.get_xticklabels():
            label.set_ha('right')

    def apply_plot_style(self, ax, grid=True, text_color=PLOT_TEXT_COLOR, transparent=False):
        """应用工业科技感绘图样式 - 干净专业"""
        if not transparent:
            ax.set_facecolor(PLOT_AX_BG)
        
        # 简化边框 - 只显示左下边框
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color(PLOT_SPINE_COLOR)
            ax.spines[spine].set_linewidth(0.8)
        
        # 刻度样式
        ax.tick_params(labelsize=PLOT_FONT_BASE, colors=text_color, 
                      direction='out', length=4, width=0.8)
        
        # 网格样式 - 轻柔辅助
        if grid:
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.4, color=PLOT_GRID_COLOR, linestyle='-', linewidth=0.5)

    def merge_intervals(self, intervals, debug=False, merge_adjacent=False):
        """合并重叠区间（可选合并相邻区间）
        :param debug: 是否打印调试信息
        :param merge_adjacent: 是否合并相邻区间（行号差1的区间）。
                               默认False，只合并真正重叠的区间（start < last_end），
                               保持不同功率值区间之间的边界。
        """
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
        merged = [intervals[0]]
        
        for start, end in intervals[1:]:
            last_start, last_end = merged[-1]
            # 判断条件：是否合并
            # - merge_adjacent=True: start <= last_end + 1 (合并重叠和相邻)
            # - merge_adjacent=False: start < last_end (只合并真正重叠的区间，不合并边界相接的)
            if merge_adjacent:
                should_merge = start <= last_end + 1
            else:
                should_merge = start < last_end  # 严格重叠才合并，不合并 start == last_end 的情况
            
            if should_merge:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        
        return merged

    def clip_intervals_to_ranges(self, intervals, ranges):
        """将区间裁剪到指定范围"""
        if not intervals:
            return []
        if not ranges:
            return self.merge_intervals(intervals)
        clipped = []
        for start, end in intervals:
            for r_start, r_end in ranges:
                if end < r_start or start > r_end:
                    continue
                clipped_start = max(start, r_start)
                clipped_end = min(end, r_end)
                if clipped_start <= clipped_end:
                    clipped.append((clipped_start, clipped_end))
        return self.merge_intervals(clipped)

    def get_predicted_intervals_for_display(self, ranges=None):
        """获取用于显示的稳态区间（按行号）- 直接返回原始区间，不进行裁剪合并"""
        base_intervals = []
        for interval in self._get_current_interval_records(allow_profile_fallback=False):
            start_line = interval.get('start_line')
            end_line = interval.get('end_line')
            if start_line is None or end_line is None:
                continue
            base_intervals.append((int(start_line), int(end_line)))
        # 直接返回原始区间，不进行裁剪和合并
        return base_intervals

    def compute_measured_avg_in_intervals(self, values, line_numbers, intervals):
        """计算实测数据在稳态区间内的平均值"""
        if values is None or line_numbers is None or not intervals:
            return None, 0
        values = np.asarray(values)
        line_numbers = np.asarray(line_numbers)
        mask = np.zeros(len(values), dtype=bool)
        for start, end in intervals:
            mask |= (line_numbers >= start) & (line_numbers <= end)
        if not mask.any():
            return None, 0
        return float(np.mean(values[mask])), int(np.sum(mask))

    def get_program_info(self, program_name):
        """获取程序信息字典"""
        if not program_name:
            return {}
        return self.sample_programs.get(program_name, {})

    def get_program_number_by_name(self, program_name):
        """按程序名获取程序号"""
        program_info = self.get_program_info(program_name)
        return program_info.get("program_number")

    def get_tool_ranges_by_id(self, program_name, tool_id):
        """按程序名+刀具号获取对齐行号范围"""
        program_info = self.get_program_info(program_name)
        return program_info.get("tools", {}).get(tool_id, [])

    def build_sample_mask(self, program_no=None, tool_ranges=None):
        """构建实测数据筛选mask（按程序号 + 刀具区间）"""
        if self.sample_data_line_numbers is None or self.sample_data_program_numbers is None:
            return None
        mask = np.ones(len(self.sample_data_line_numbers), dtype=bool)
        if program_no is not None:
            mask &= (self.sample_data_program_numbers == program_no)
        if tool_ranges:
            range_mask = np.zeros(len(self.sample_data_line_numbers), dtype=bool)
            for r_start, r_end in tool_ranges:
                range_mask |= (self.sample_data_line_numbers >= r_start) & (self.sample_data_line_numbers <= r_end)
            mask &= range_mask
        return mask

    def build_process_mask(self, line_numbers, tool_ranges=None):
        """构建工艺信息筛选mask（按刀具区间）。"""
        if line_numbers is None:
            return None
        line_arr = np.asarray(line_numbers)
        mask = np.ones(len(line_arr), dtype=bool)
        if tool_ranges:
            mask = np.zeros(len(line_arr), dtype=bool)
            for r_start, r_end in tool_ranges:
                mask |= (line_arr >= r_start) & (line_arr <= r_end)
        return mask

    def count_points_by_blocks(self, mask, mode="sum"):
        """按连续区块统计点数。"""
        blocks = self.compute_contiguous_blocks(mask)
        if not blocks:
            return 0, []
        counts = [end - start + 1 for start, end in blocks]
        if mode == "max_block":
            return max(counts), blocks
        return sum(counts), blocks

    def compute_tool_measured_mean(self, program_name, tool_id):
        """计算指定程序+刀具在稳态区间内的实测均值"""
        if not self.sample_data_loaded or self.sample_data_values is None:
            return None, 0, []
        program_no = self.get_program_number_by_name(program_name)
        tool_ranges = self.get_tool_ranges_by_id(program_name, tool_id)
        intervals = self.get_predicted_intervals_for_display(tool_ranges)
        if not intervals:
            return None, 0, intervals
        mask = self.build_sample_mask(program_no, tool_ranges)
        if mask is None or not mask.any():
            return None, 0, intervals
        source_idx = int(self.sample_data_source.get())
        values = self.sample_data_values[:, source_idx][mask]
        line_numbers = self.sample_data_line_numbers[mask]
        mean_val, count = self.compute_measured_avg_in_intervals(values, line_numbers, intervals)
        return mean_val, count, intervals

    def format_line_point(self, line_number, point_index):
        """格式化行点：行号.点序号"""
        try:
            ln = int(line_number)
            pt = max(int(point_index) + 1, 1)
        except Exception:
            return f"{line_number}.{point_index}"
        return f"{ln}.{pt}"

    def collect_line_point_intervals_for_tool(self, program_name, tool_id):
        """获取指定程序+刀具的稳态区间行点范围（包含每个区间的平均值）
        
        使用工艺信息索引的精确x坐标边界来匹配SampleData点，
        确保与图表绘制的区间边界完全一致。
        """
        if not self.sample_data_loaded or self.sample_data_line_numbers is None:
            return []
        interval_records = self._get_current_interval_records(allow_profile_fallback=False)
        if not interval_records:
            return []
        if not self.data:
            return []

        program_no = self.get_program_number_by_name(program_name)
        tool_ranges = self.get_tool_ranges_by_id(program_name, tool_id)
        if not tool_ranges:
            return []
        
        base_mask = self.build_sample_mask(program_no, tool_ranges)
        if base_mask is None or not base_mask.any():
            return []
        
        # 获取SampleData的数据
        sample_line_numbers = np.asarray(self.sample_data_line_numbers)
        sample_point_indices = np.asarray(self.sample_data_point_indices)
        sample_x_positions = self.sample_data_x_positions
        if sample_x_positions is None:
            sample_x_positions = np.asarray(
                self.compute_line_x_positions(self.sample_data_line_numbers, blocks=self.sample_data_base_blocks)
            )
        else:
            sample_x_positions = np.asarray(sample_x_positions)
        
        # 计算工艺信息数据的区间边界（与图表绘制相同）
        process_line_numbers = [d.get('line_no_aligned', i) for i, d in enumerate(self.data)]
        process_blocks = self.compute_sequence_blocks(process_line_numbers)
        process_start_bounds, process_end_bounds = self.compute_line_segment_bounds(
            process_line_numbers,
            blocks=process_blocks
        )
        
        # 获取实测数据值用于计算平均值
        data_source_idx = int(self.sample_data_source.get())
        if self.sample_data_values is not None and len(self.sample_data_values.shape) > 1:
            sample_values = self.sample_data_values[:, data_source_idx]
        else:
            sample_values = self.sample_data_values
        
        interval_strings = []
        
        for interval in interval_records:
            start_idx_iv = interval.get('start_idx')
            end_idx_iv = interval.get('end_idx')
            start_line = interval.get('start_line')
            end_line = interval.get('end_line')
            
            if start_idx_iv is None or end_idx_iv is None:
                continue
            if start_line is None or end_line is None:
                continue
            
            # 检查是否在当前刀具范围内
            in_range = False
            for r_start, r_end in tool_ranges:
                if not (end_line < r_start or start_line > r_end):
                    in_range = True
                    break
            if not in_range:
                continue
            
            # 获取工艺信息区间的边界（与图表绘制一致）
            display_start_x = interval.get("display_start_x")
            display_end_x = interval.get("display_end_x")
            if np.isfinite(display_start_x) and np.isfinite(display_end_x) and float(display_end_x) > float(display_start_x):
                interval_start_x = float(display_start_x)
                interval_end_x = float(display_end_x)
            else:
                if start_idx_iv >= len(process_start_bounds) or end_idx_iv >= len(process_end_bounds):
                    continue
                interval_start_x = process_start_bounds[start_idx_iv]
                interval_end_x = process_end_bounds[end_idx_iv]
            
            # 在SampleData中找x坐标落在区间内的点
            # 使用半开区间 [start_x, end_x)
            mask = base_mask & (sample_x_positions >= interval_start_x) & (sample_x_positions < interval_end_x)
            if not mask.any():
                continue

            for block_start, block_end in self.compute_contiguous_blocks(mask):
                first_idx = int(block_start)
                last_idx = int(block_end)

                start_lp = self.format_line_point(sample_line_numbers[first_idx], sample_point_indices[first_idx])
                end_lp = self.format_line_point(sample_line_numbers[last_idx], sample_point_indices[last_idx])

                interval_values = sample_values[block_start:block_end + 1]
                finite_mask = np.isfinite(interval_values)
                if finite_mask.any():
                    avg_value = float(np.mean(interval_values[finite_mask]))
                else:
                    avg_value = 0.0

                interval_strings.append(f"{start_lp}-{end_lp}:{avg_value:.6f}")
        
        return interval_strings

    def draw_tool_mean_ideal_lines(self, ax, tool_ranges, mean_val, ideal_val, color, label_prefix=None, display_spans=None):
        """在指定刀具区间范围绘制均值/理想值线"""
        if mean_val is None or not tool_ranges:
            return []
        spans = list(display_spans) if display_spans else [(start, end + 1) for start, end in tool_ranges]
        if not spans:
            return []
        ideal_lines = []
        for start, end in spans:
            ax.plot([start, end], [mean_val, mean_val],
                    color=color, linewidth=0.9, alpha=0.22, zorder=6)
            if ideal_val is not None:
                line, = ax.plot([start, end], [ideal_val, ideal_val],
                                color=color, linestyle='--', linewidth=0.95, alpha=0.36, zorder=6)
                ideal_lines.append(line)
        if label_prefix:
            last_start, last_end = spans[-1]
            x_offset = max((last_end - last_start) * 0.03, 0.3)
            ax.text(last_end + x_offset, mean_val, f"{label_prefix}均值:{mean_val:.3f}",
                    fontsize=PLOT_FONT_BASE, color=color, va='center')
            if ideal_val is not None:
                ax.text(last_end + x_offset, ideal_val, f"{label_prefix}理想:{ideal_val:.3f}",
                        fontsize=PLOT_FONT_BASE, color=color, va='center')
        return ideal_lines
