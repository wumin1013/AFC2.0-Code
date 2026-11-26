"""Research-only Tkinter integration. Workers never access Tk widgets."""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from dataclasses import asdict, replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import (BasePowerModel, CalibrationConfig, FeedPlan, MachineConfig, ModelConfig,
              build_load_models, fit_base_power, identify_events, inspect_dataset, load_models,
              parse_nc, plan_feed, predict_base_from_commands, predict_base_from_measured_speed,
              predict_cutting_power, prepare_load_samples, read_measurement, PositionMemory, predict_base_event)
from .base_reference import (IdleReference, calibrate_idle_reference, validate_reference_session,
                             same_reference_source, verify_reference_record)
from .base_greybox import GreyBoxConfig, GreyBoxModel, predict_greybox_event
from .greybox_validation import compare_greybox_models
from .base_recovery import RecoveryModel, initialize_recovery_reference, predict_recovery_event
from .recovery_validation import compare_recovery_models
from .plotting import configure_chinese_plots, MODEL_LABELS
from .common import Cancelled, checkpoint, json_value, save_json
from .samples import read_process_table, prepare_forward_features, PHYSICS_COLUMNS
from .training import train_two_stage


class LoadPredictionWorkbench(ttk.Frame):
    def __init__(self, master, output_dir):
        super().__init__(master)
        configure_chinese_plots()
        self.output_dir = Path(output_dir)
        self.records, self.events, self.base, self.feed_plan, self.bundle = [], [], None, None, None
        self.base_candidate = None
        self.idle_reference = None
        self.use_idle_reference = tk.BooleanVar(value=False)
        self.reference_status = tk.StringVar(value="未启用空载参考")
        self.greybox_models, self.greybox_report = {}, None
        self.greybox_config = GreyBoxConfig()
        self.recovery_models, self.recovery_report = {}, None
        self.machine, self.calibration, self.model_config = MachineConfig(), CalibrationConfig(), ModelConfig()
        self.nc_path, self.samples = None, None
        self.forward_features = None
        self.memory, self.last_prediction = PositionMemory(), None
        self.jobs, self.cancel_event, self.busy = queue.Queue(), threading.Event(), False
        self.buttons = []
        self.status = tk.StringVar(value="待导入数据；负载网络尚未训练")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill=tk.X, padx=6, pady=5)
        self.pages = {}
        for name in ("数据检查", "基础功率", "指令进给", "负载模型"):
            page = ttk.Frame(self.tabs, padding=8)
            self.tabs.add(page, text=name)
            self.pages[name] = page
        self._build_controls()
        row = ttk.Frame(self)
        row.pack(fill=tk.X, padx=8)
        ttk.Label(row, textvariable=self.status).pack(side=tk.LEFT)
        ttk.Button(row, text="取消当前任务", command=self.cancel).pack(side=tk.RIGHT)
        pane = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=5)
        text_frame, plot_frame = ttk.Frame(pane), ttk.Frame(pane)
        pane.add(text_frame, weight=1)
        pane.add(plot_frame, weight=2)
        self.report = tk.Text(text_frame, width=40, height=10, wrap="word", font=("Microsoft YaHei UI", 9))
        scroll = ttk.Scrollbar(text_frame, command=self.report.yview)
        self.report.configure(yscrollcommand=scroll.set)
        self.report.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.figure = Figure(figsize=(5, 3), dpi=90)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.bind("<Destroy>", self._destroy, add="+")
        self.poll_id = self.after(100, self._poll)

    def _button(self, page, text, command, row=0, column=0):
        def guarded():
            try:
                command()
            except Exception as exc:
                self.status.set("输入检查未通过")
                self.show(str(exc))
                messagebox.showerror("负载预测", str(exc), parent=self)
        button = ttk.Button(page, text=text, command=guarded)
        button.grid(row=row, column=column, padx=4, pady=4, sticky="w")
        self.buttons.append(button)
        return button

    def _build_controls(self):
        page = self.pages["数据检查"]
        self._button(page, "检查实验目录", self._inspect)
        self._button(page, "查看采样曲线", self._view_csv, column=1)
        self.deep = tk.BooleanVar(value=False)
        ttk.Checkbutton(page, text="逐点检查全部文件（较慢）", variable=self.deep).grid(row=0, column=2)
        ttk.Label(page, text="原始CSV只读；通道单位、异常和文件配套情况分别检查。").grid(row=1, column=0, columnspan=4, sticky="w")
        page = self.pages["基础功率"]
        for col, (text, fn) in enumerate((("导入空载CSV", self._load_idle), ("审核平台/指令", self._edit_events),
                                         ("标定设置", self._edit_calibration), ("训练/标定", self._fit_base))):
            self._button(page, text, fn, column=col)
        for col, (text, fn) in enumerate((("保存候选/当前模型", self._save_base), ("加载基础模型", self._load_base),
                                         ("查看扣除结果", self._subtract), ("加工前基础功率", self._forecast_base))):
            self._button(page, text, fn, row=1, column=col)
        self._button(page, "转速查询：峰值/时间", self._query_base_event, row=2)
        ttk.Label(page, text="从已稳定停转启动；时间相对观测响应起点，控制器延迟未标定。").grid(row=2, column=1, columnspan=3, sticky="w")
        for col, (text, fn) in enumerate((("导入空载参考（可选）", self._import_idle_reference),
                                         ("清除参考", self._clear_idle_reference),
                                         ("保存参考", self._save_idle_reference), ("加载参考（同会话）", self._load_idle_reference))):
            self._button(page, text, fn, row=3, column=col)
        for col, (text, fn) in enumerate((("一阶/二阶联合辨识对比", self._compare_greybox),
                                         ("灰箱候选转速查询", self._query_greybox),
                                         ("加载灰箱候选", self._load_greybox), ("灰箱标定设置", self._edit_greybox))):
            self._button(page, text, fn, row=4, column=col)
        ttk.Label(page, text="灰箱结果单独保存为研究候选；报告区分实测转速、此前空载初始化与加工前估计。").grid(row=5, column=0, columnspan=4, sticky="w")
        for col, (text, fn) in enumerate((("恢复尾段对比", self._compare_recovery),
                                         ("恢复候选查询", self._query_recovery), ("加载恢复候选", self._load_recovery))):
            self._button(page, text, fn, row=6, column=col)
        toggle = ttk.Checkbutton(page, text="应用当前空载参考", variable=self.use_idle_reference,
                                 command=self._reference_choice_changed)
        toggle.grid(row=7, column=0, sticky="w", padx=4)
        self.buttons.append(toggle)
        self._button(page, "参考会话设置", self._edit_reference_session, row=7, column=1)
        ttk.Label(page, textvariable=self.reference_status).grid(row=7, column=2, columnspan=3, sticky="w")
        page = self.pages["指令进给"]
        for col, (text, fn) in enumerate((("机床参数", self._edit_machine), ("导入配置", self._load_machine),
                                         ("保存配置", self._save_machine), ("选择/解析NC", self._parse_nc))):
            self._button(page, text, fn, column=col)
        for col, (text, fn) in enumerate((("生成指令进给", self._plan), ("保存运动计划", self._save_plan), ("加载运动计划", self._load_plan))):
            self._button(page, text, fn, row=1, column=col)
        page = self.pages["负载模型"]
        for col, (text, fn) in enumerate((("模型配置", self._edit_model), ("检查模型结构", self._build_models),
                                         ("准备单轮样本", self._prepare), ("导出候选样本", self._export_samples))):
            self._button(page, text, fn, column=col)
        self._button(page, "保存模型结构", self._save_models, row=1)
        self._button(page, "加载模型包", self._load_models, row=1, column=1)
        self._button(page, "准备加工前特征", self._prepare_forward, row=0, column=4)
        self._button(page, "统一训练与新案例预测", self._open_unified_workbench, row=1, column=2)
        self._button(page, "预测功率分解", self._predict_load, row=2)
        self._button(page, "更新已完成轮次记忆", self._update_memory, row=2, column=1)
        self._button(page, "保存/加载记忆", self._memory_dialog, row=2, column=2)
        self._button(page, "重置记忆", self._reset_memory, row=2, column=3)

    def _open_unified_workbench(self):
        """Desktop opens the same service; no protocol or training logic is copied."""
        import os
        import subprocess
        import sys
        import time
        import urllib.request
        import webbrowser
        from .training_service import TrainingService
        service = TrainingService(self.output_dir / "service")
        url = "http://127.0.0.1:8765"
        def connected():
            try:
                with urllib.request.urlopen(url + "/api/v1/info", timeout=.4) as response:
                    info = json.loads(response.read().decode("utf-8"))
                    if info.get("protocol_version") != service.protocol()["version"] or Path(info.get("output_dir", "")).resolve() != service.output_dir:
                        raise RuntimeError("8765端口已连接其他结果目录；请关闭该服务或使用相同output-dir")
                    return True
            except (OSError, ValueError, KeyError):
                return False
        def operation(cancel):
            if not connected():
                env = dict(os.environ)
                env.update(PYTHONUTF8="1", MKL_THREADING_LAYER="SEQUENTIAL", OMP_NUM_THREADS="1", MPLBACKEND="Agg")
                env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
                with (service.output_dir / "web_service.log").open("ab") as log:
                    process = subprocess.Popen([sys.executable, "-m", "spindle_power.load_prediction.web_service", "--output-dir", str(service.output_dir)],
                        env=env, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                for _ in range(40):
                    checkpoint(cancel)
                    if connected():
                        break
                    if process.poll() is not None:
                        raise ValueError("本地服务启动失败，详情见 web_service.log")
                    time.sleep(.1)
                else:
                    raise ValueError("本地服务未在限定时间内启动")
            return dict(url=url, protocol=service.protocol(), output_dir=str(service.output_dir))
        def done(result):
            self.show(result)
            webbrowser.open(result["url"])
        self.start("连接统一训练与预测服务…", operation, done)

    def show(self, value):
        self.report.delete("1.0", tk.END)
        self.report.insert(tk.END, value if isinstance(value, str) else json.dumps(json_value(value), ensure_ascii=False, indent=2))

    def plot(self, x, series, xlabel="时间（s）", ylabel="功率（W）"):
        self.axes.clear()
        stride = max(1, len(x) // 15000)
        for label, values in series.items():
            self.axes.plot(np.asarray(x)[::stride], np.asarray(values)[::stride], label=label, linewidth=.8)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.axes.legend(fontsize=8)
        self.axes.grid(alpha=.2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def start(self, label, operation, done):
        if self.busy:
            raise ValueError("已有任务正在运行")
        self.busy = True
        self.cancel_event = threading.Event()
        cancel = self.cancel_event
        self.status.set(label)
        for button in self.buttons:
            button.configure(state="disabled")
        def worker():
            try:
                result = operation(cancel.is_set)
                checkpoint(cancel.is_set)
                self.jobs.put((done, result, None))
            except Exception as exc:
                self.jobs.put((done, None, exc))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            done, result, error = self.jobs.get_nowait()
        except queue.Empty:
            pass
        else:
            self.busy = False
            for button in self.buttons:
                button.configure(state="normal")
            if self.cancel_event.is_set() or isinstance(error, Cancelled):
                self.status.set("已取消；原有效结果保留")
            elif error:
                self.status.set("任务失败；原有效结果保留")
                self.show(str(error))
            else:
                try:
                    self.status.set("任务完成")
                    done(result)
                except Exception as exc:
                    self.status.set("结果显示失败")
                    self.show(str(exc))
        if self.winfo_exists():
            self.poll_id = self.after(100, self._poll)

    def cancel(self):
        self.cancel_event.set()
        if self.busy:
            self.status.set("正在取消…")

    def _destroy(self, event):
        if event.widget is self:
            self.cancel_event.set()
            if getattr(self, "poll_id", None):
                try:
                    self.after_cancel(self.poll_id)
                except tk.TclError:
                    pass

    def _choose_save(self, title, extension=".json"):
        return filedialog.asksaveasfilename(parent=self, title=title, initialdir=str(self.output_dir), defaultextension=extension)

    def _json_editor(self, title, value, accept):
        dialog = tk.Toplevel(self)
        dialog.title(title)
        dialog.geometry("760x550")
        editor = tk.Text(dialog, wrap="none", font=("Consolas", 11))
        editor.pack(fill=tk.BOTH, expand=True)
        editor.insert("1.0", json.dumps(json_value(value), ensure_ascii=False, indent=2))
        def save():
            try:
                accept(json.loads(editor.get("1.0", tk.END)))
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror(title, str(exc), parent=dialog)
        ttk.Button(dialog, text="保存设置", command=save).pack(pady=6)

    def _inspect(self):
        path = filedialog.askdirectory(parent=self, title="选择实验数据根目录")
        if path:
            deep = self.deep.get()
            def done(result):
                self.show(result)
                save_json(self.output_dir / "dataset_report.json", result)
            self.start("正在检查实验目录…", lambda c: inspect_dataset(path, deep=deep, cancel=c), done)

    def _view_csv(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Scope CSV", "*.csv")])
        if path:
            def done(record):
                self.show({"source": record.source, "samples": len(record.time_s), **record.metadata})
                record.require("spindle_power_w")
                self.plot(record.time_s, {"实测功率": record.values["spindle_power_w"]})
            self.start("读取采样文件…", lambda c: read_measurement(path, cancel=c), done)

    def _load_idle(self):
        paths = filedialog.askopenfilenames(parent=self, filetypes=[("空载CSV", "*.csv")])
        if not paths:
            return
        config = self.calibration
        def operation(cancel):
            records = [read_measurement(path, cancel=cancel) for path in paths]
            return records, [identify_events(r, config) for r in records]
        def done(result):
            self.records, self.events = result
            self.show({"files": [r.source for r in self.records], "events": self.events, "指令来源": "estimated，可在审核窗口修正"})
            self.plot(self.records[0].time_s, {"实测功率": self.records[0].values["spindle_power_w"]})
        self.start("读取空载文件并识别平台…", operation, done)

    def _edit_events(self):
        if not self.events:
            raise ValueError("先导入空载CSV")
        def accept(value):
            from .base_power import _event_arrays
            if len(value) != len(self.records):
                raise ValueError("事件列表数量必须与文件数量一致")
            for record, events in zip(self.records, value):
                _event_arrays(record, events)
            self.events = value
        self._json_editor("审核事件：start/end 为稳态窗口，command_time 为估计指令时刻，单位秒", self.events, accept)

    def _edit_calibration(self):
        def accept(value):
            config = CalibrationConfig(**value)
            config.validate()
            self.calibration = config
        self._json_editor("标定参数（更改后重新导入以重识别事件）", asdict(self.calibration), accept)

    def _fit_base(self):
        if len(self.records) < 2:
            raise ValueError("请导入至少两份空载CSV，当前实验建议三份")
        records, events, config = self.records, self.events, self.calibration
        def done(model):
            self.show(model.report)
            self.base_candidate = model
            self._set_idle_reference(None)
            had_valid_model = self.base is not None and self.base.status == "validated_in_sample_domain"
            if model.status == "validated_in_sample_domain" or not had_valid_model:
                self.base = model
            save_json(self.output_dir / "event_v2" / "base_validation_report.json", model.report)
            result = predict_base_from_measured_speed(records[0].time_s, records[0].values["spindle_speed_rpm"], model)
            self.plot(records[0].time_s, {"实测功率": records[0].values["spindle_power_w"], "基础功率模型": result["power_w"], "仅稳态曲线": result["steady_w"]})
            self.status.set("标定通过（限已覆盖工况）" if model.status == "validated_in_sample_domain" else
                            "候选待验证；已有有效模型保留" if had_valid_model else "标定待验证；可查询和保存候选")
        self.start("空载标定与整文件留出验证…", lambda c: fit_base_power(records, events=events, config=config, cancel=c), done)

    def _edit_greybox(self):
        def accept(value):
            candidate = GreyBoxConfig(**value)
            candidate.validate()
            self.greybox_config = candidate
        self._json_editor("灰箱标定设置", asdict(self.greybox_config), accept)

    def _compare_greybox(self):
        if len(self.records) < 3:
            raise ValueError("联合辨识对比需要至少三份独立空载CSV")
        records, events, config, fitting = self.records, self.events, self.calibration, self.greybox_config
        def done(result):
            output = self.output_dir / "greybox_validation"
            for name, model in result["models"].items():
                model.save(output / f"{name}.json")
            save_json(output / "comparison.json", result["report"])
            self.greybox_models, self.greybox_report = result["models"], result["report"]
            self.show({"事件模型基准": result["report"]["baseline"]["summary"],
                       "灰箱对比": {k: v["summary"] for k, v in result["report"]["variants"].items()},
                       "结论": result["report"]["decision"], "适用边界": result["report"]["limits"],
                       "完整报告目录": str(output)})
            self.status.set("灰箱对比完成；候选单独保存，当前基础模型保持原选择")
        self.start("一阶/二阶联合辨识与整文件留出验证…",
                   lambda c: compare_greybox_models(records, config=config, fitting=fitting, events=events, cancel=c), done)

    def _load_greybox(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("灰箱候选JSON", "*.json")])
        if path:
            model = GreyBoxModel.load(path)
            self.greybox_models[f"order{model.order}_{model.drive}"] = model
            self.show({"状态": model.status, "时间常数 s": model.taus_s, "报告": model.report})
            self.status.set("灰箱研究候选已加载；可单独查询")

    def _query_greybox(self):
        if not self.greybox_models:
            raise ValueError("先完成一阶/二阶对比或加载灰箱候选")
        default = "order2_inertial" if "order2_inertial" in self.greybox_models else next(iter(self.greybox_models))
        def accept(value):
            options = dict(value)
            name = options.pop("model")
            if name not in self.greybox_models:
                raise ValueError(f"可选模型：{list(self.greybox_models)}")
            model = self.greybox_models[name]
            def done(result):
                self.plot(result["time_s"], {MODEL_LABELS[name]: result["power_w"], "稳态曲线": result["steady_w"]})
                self.show({"模型": name, "状态": result["status"], "事件特征": result["characteristics"],
                           "转速来源": result["speed_source"], "时间基准": result["timing"], "适用边界": result["warnings"]})
            self.start("灰箱候选转速查询…", lambda c: predict_greybox_event(model=model, cancel=c, **options), done)
        self._json_editor("灰箱候选查询（model可选：" + ", ".join(self.greybox_models) + "）",
                          dict(model=default, target_speed_rpm=10000., initial_speed_rpm=0., sample_period_s=.001), accept)

    def _compare_recovery(self):
        if len(self.records) < 3:
            raise ValueError("恢复对比需要至少三份独立空载CSV")
        records, events, config = self.records, self.events, self.calibration
        def done(result):
            output = self.output_dir / "recovery_validation"
            for name, model in result["models"].items():
                model.save(output / f"{name}.json")
            save_json(output / "comparison.json", result["report"])
            self.recovery_models, self.recovery_report = result["models"], result["report"]
            self.show({"恢复对比": {k: {"候选": v["summary"], "同口径基准": v["baseline_summary"],
                                      "文件与转速交叉留出": v["crossed_speed_summary"]}
                                    for k, v in result["report"]["variants"].items()},
                       "结论": result["report"]["decision"], "边界": result["report"]["limits"], "报告目录": str(output)})
            self.status.set("恢复对比完成；候选单独保存")
        self.start("恢复尾段拟合与文件/档位留出对比…",
                   lambda c: compare_recovery_models(records, events=events, config=config, cancel=c), done)

    def _load_recovery(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("恢复候选JSON", "*.json")])
        if path:
            model = RecoveryModel.load(path)
            self.recovery_models[model.method] = model
            self.show({"状态": model.status, "参数": model.coefficients, "报告": model.report})
            self.status.set("恢复研究候选已加载；可单独查询")

    def _query_recovery(self):
        if not self.recovery_models:
            raise ValueError("先完成恢复对比或加载恢复候选")
        def accept(value):
            options = dict(value)
            name = options.pop("model")
            if name not in self.recovery_models:
                raise ValueError(f"可选模型：{list(self.recovery_models)}")
            model = self.recovery_models[name]
            index = options.pop("reference_record_index")
            end = options.pop("reference_end_s")
            confirmed = options.pop("no_load_confirmed")
            record = None
            if index is not None:
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.records):
                    raise ValueError("参考文件序号须为已导入空载文件的从0开始序号")
                if end is None or confirmed is not True:
                    raise ValueError("使用参考须填写已完成时刻，并确认此前整个前缀为空载")
                record = self.records[index]
            def operation(cancel):
                reference = initialize_recovery_reference(record, model, end_s=end, no_load_confirmed=confirmed, cancel=cancel) if record is not None else None
                return predict_recovery_event(model=model, reference=reference, source=record.source if record is not None else None, cancel=cancel, **options)
            def done(result):
                self.plot(result["time_s"], {MODEL_LABELS[name]: result["power_w"]})
                self.show({"状态": result["status"], "事件特征": result["characteristics"], "恢复参数/s": result["recovery_parameter_s"],
                           "参考": result["reference"], "时间尺度": result["time_multiplier"], "时间基准": result["timing"], "边界": result["warnings"]})
            self.start("恢复候选转速查询…", operation, done)
        self._json_editor("恢复查询；reference_record_index=null表示无参考；onset_s为同记录时间轴上的未来响应起点",
                          dict(model=next(iter(self.recovery_models)), target_speed_rpm=10000., initial_speed_rpm=0.,
                               sample_period_s=.001, reference_record_index=None, reference_end_s=None,
                               no_load_confirmed=False, onset_s=None), accept)

    def _save_base(self):
        model = self.base_candidate or self.base
        if model is None:
            raise ValueError("没有基础模型")
        path = self._choose_save("保存基础模型")
        if path:
            model.save(path)

    def _load_base(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("模型JSON", "*.json")])
        if path:
            self.base = BasePowerModel.load(path)
            self.base_candidate = self.base
            self._set_idle_reference(None)
            self.show(asdict(self.base))
            self.status.set(self.base.status)

    def _query_base_event(self):
        model = self.base_candidate or self.base
        if model is None:
            raise ValueError("先标定或加载基础模型")
        def accept(value):
            def done(result):
                self.plot(result["time_s"], {"事件基础功率": result["power_w"]})
                c = result["characteristics"]
                self.axes.scatter([c["peak_time_s"]], [c["extremum_power_w"]], color="red", s=24)
                if c["settling_time_s"] is not None:
                    self.axes.axvline(c["settling_time_s"], color="green", linestyle="--", label="进入稳态")
                self.axes.legend(fontsize=8)
                self.canvas.draw_idle()
                self.show({"模型状态": result["status"], "初始转速 r/min": c["initial_speed_rpm"],
                           "目标转速 r/min": c["target_speed_rpm"],
                           "峰值/制动谷值 W": c["extremum_power_w"], "达峰/谷时间 s": c["peak_time_s"],
                           "最终稳态功率 W": c["steady_power_w"], "进入稳态时间 s": c["settling_time_s"],
                           "峰值后恢复稳态耗时 s": c["peak_to_settle_s"],
                           "各指标留出验收": result["validation_components"],
                           "空载参考校正": result["reference_adjustment"],
                           "参考应用范围": result.get("reference_usage"),
                           "仅转速输入的时间重复性": result["repeatability"],
                           "标定事件实测范围（非置信区间）": result["observed_ranges"],
                           "时间基准": result["timing"], "适用条件": result["assumptions"]})
            def launch(reference, session):
                self.start("生成单次主轴事件曲线…", lambda cancel: predict_base_event(
                    model=model, reference=reference, session=session, record_source="单次事件查询",
                    onset_s=0. if reference is not None else None, cancel=cancel, **value), done)
            self._with_idle_reference(model, "单次事件查询", launch)
        self._event_query_form(accept)

    def _import_idle_reference(self):
        model = self.base_candidate or self.base
        if model is None:
            raise ValueError("先标定或加载基础模型")
        path = filedialog.askopenfilename(parent=self, title="选择当前会话、加工前已完成的空载CSV", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        def done(record):
            events = [e for e in identify_events(record) if e["speed_rpm"] > 0 and e["end_s"] - e["start_s"] >= 1.]
            if not events:
                raise ValueError("未找到至少1秒的正转速稳定平台")
            event = events[0]
            dialog = tk.Toplevel(self)
            dialog.title("选择已完成的稳定空载参考窗口")
            platform = ttk.Combobox(dialog, state="readonly", width=47,
                                   values=[f'{e["speed_rpm"]:g} r/min，{e["start_s"]:.3f}—{e["end_s"]:.3f} s' for e in events])
            platform.current(0)
            ttk.Label(dialog, text="参考平台").grid(row=0, column=0, padx=10, pady=6)
            platform.grid(row=0, column=1, padx=10, pady=6)
            start = tk.StringVar(value=str(round(event["end_s"] - 1., 6)))
            end = tk.StringVar(value=str(round(event["end_s"], 6)))
            def select_platform(_event):
                chosen = events[platform.current()]
                start.set(str(round(chosen["end_s"] - 1., 6)))
                end.set(str(round(chosen["end_s"], 6)))
            platform.bind("<<ComboboxSelected>>", select_platform)
            for row, (name, value) in enumerate((("窗口起点 s", start), ("窗口终点 s", end)), 1):
                ttk.Label(dialog, text=name).grid(row=row, column=0, padx=10, pady=6)
                ttk.Entry(dialog, textvariable=value).grid(row=row, column=1, padx=10, pady=6)
            context_fields = self._session_fields(dialog, 3, dict(time_origin_s=0.))
            confirmed = tk.BooleanVar(value=False)
            ttk.Checkbutton(dialog, text="确认：所填机床/会话的已完成空载窗口，不含切削", variable=confirmed).grid(row=6, column=0, columnspan=2, padx=10, pady=8)
            def submit():
                try:
                    a, b = float(start.get()), float(end.get())
                    if not confirmed.get():
                        raise ValueError("请确认窗口为空载，不能用切削功率修正基础功率")
                    session = self._read_session_fields(context_fields)
                    def accept(reference):
                        current = self.base_candidate or self.base
                        reference.validate(current)
                        self._set_idle_reference(reference)
                        self.show({"参考": asdict(reference), "用途": "同会话后续查询、整段预测、扣除与候选样本；校正状态仍待验证"})
                    self.start("提取空载参考…", lambda c: calibrate_idle_reference(record, model, start_s=a, end_s=b,
                               no_load_confirmed=True, session=session, cancel=c), accept)
                    dialog.destroy()
                except Exception as exc:
                    messagebox.showerror("空载参考", str(exc), parent=dialog)
            ttk.Button(dialog, text="提取参考", command=submit).grid(row=7, column=0, columnspan=2, pady=8)
        self.start("读取空载参考文件…", lambda c: read_measurement(path, cancel=c), done)

    def _clear_idle_reference(self):
        self._set_idle_reference(None)
        self.status.set("已清除空载参考；恢复为仅转速估计")

    def _set_idle_reference(self, reference):
        self.idle_reference = reference
        self.use_idle_reference.set(reference is not None and reference.session is not None)
        self._reference_choice_changed()

    def _reference_choice_changed(self):
        self.samples = self.last_prediction = None
        reference = self.idle_reference
        if not self.use_idle_reference.get():
            self.reference_status.set("参考未启用" if reference is None or reference.session is not None else "旧参考缺少会话信息，请设置或重新提取")
        elif reference is None or reference.session is None:
            self.use_idle_reference.set(False)
            self.reference_status.set("需先提取参考并填写会话信息")
        else:
            self.reference_status.set(f'参考已启用：{reference.session["machine_id"]} / {reference.session["session_id"]}（候选）')

    @staticmethod
    def _session_fields(dialog, first_row, defaults):
        fields = {}
        for row, (key, label) in enumerate((("machine_id", "机床标识"), ("session_id", "连续运行会话标识"),
                                           ("time_origin_s", "本文件/事件0秒对应会话时刻（s）")), first_row):
            fields[key] = tk.StringVar(value=str(defaults.get(key, "")))
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=10, pady=6)
            ttk.Entry(dialog, textvariable=fields[key], width=34).grid(row=row, column=1, padx=10, pady=6)
        return fields

    @staticmethod
    def _read_session_fields(fields):
        value = {key: variable.get().strip() for key, variable in fields.items()}
        try:
            value["time_origin_s"] = float(value["time_origin_s"])
        except ValueError as exc:
            raise ValueError("需填写时间原点；不同CSV的0秒不能直接视为同一时刻") from exc
        return validate_reference_session(value)

    def _edit_reference_session(self):
        reference = self.idle_reference
        if reference is None:
            raise ValueError("先提取或加载空载参考")
        dialog = tk.Toplevel(self)
        dialog.title("参考来源的会话信息")
        fields = self._session_fields(dialog, 0, reference.session or {})
        ttk.Label(dialog, text="按真实记录填写；重启或改变主轴设置后应重新提取参考。", wraplength=530).grid(row=3, column=0, columnspan=2, padx=10, pady=8)
        def submit():
            try:
                candidate = replace(reference, session=self._read_session_fields(fields))
                candidate.validate(self.base_candidate or self.base)
                self._set_idle_reference(candidate)
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("参考会话", str(exc), parent=dialog)
        ttk.Button(dialog, text="保存会话信息", command=submit).grid(row=4, column=0, columnspan=2, pady=8)

    def _with_idle_reference(self, model, source, accept):
        """Bind each new target explicitly; unrelated file clocks start at zero."""
        if not self.use_idle_reference.get():
            accept(None, None)
            return
        reference = self.idle_reference
        if reference is None:
            raise ValueError("没有可用空载参考")
        reference.validate(model)
        session = validate_reference_session(reference.session)
        if same_reference_source(source, reference):
            accept(reference, session)
            return
        dialog = tk.Toplevel(self)
        dialog.title("目标数据与空载参考的时间关系")
        fields = self._session_fields(dialog, 0, {k: session[k] for k in ("machine_id", "session_id")})
        end = session["time_origin_s"] + reference.end_s
        ttk.Label(dialog, text=f"参考已完成时刻：会话第 {end:.3f} 秒。\n目标：{source}\n请填写目标0秒在同一会话时间轴上的时刻，须晚于参考完成。",
                  wraplength=600).grid(row=3, column=0, columnspan=2, padx=10, pady=8)
        confirmed = tk.BooleanVar(value=False)
        ttk.Checkbutton(dialog, text="确认同机床、同一连续会话，期间未重启或改变主轴设置", variable=confirmed).grid(row=4, column=0, columnspan=2, padx=10, pady=8)
        def submit():
            try:
                target = self._read_session_fields(fields)
                if not confirmed.get():
                    raise ValueError("请核实目标与参考属于同一连续运行会话")
                from .base_reference import reference_use_mask
                reference_use_mask([0.], model, reference, session=target, record_source=source)
                accept(reference, target)
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("参考应用", str(exc), parent=dialog)
        ttk.Button(dialog, text="应用参考并继续", command=submit).grid(row=5, column=0, columnspan=2, pady=8)

    def _save_idle_reference(self):
        if self.idle_reference is None:
            raise ValueError("没有空载参考")
        path = self._choose_save("保存同会话空载参考")
        if path:
            self.idle_reference.save(path)

    def _load_idle_reference(self):
        model = self.base_candidate or self.base
        if model is None:
            raise ValueError("先标定或加载基础模型")
        path = filedialog.askopenfilename(parent=self, title="仅加载当前机床同一会话的空载参考", filetypes=[("JSON", "*.json")])
        if path:
            reference = IdleReference.load(path)
            reference.validate(model)
            self._set_idle_reference(reference)
            self.show({"参考": asdict(reference), "用途": "仅适用于参考完成后的同会话数据；旧参考需补齐会话信息"})

    def _event_query_form(self, accept):
        dialog = tk.Toplevel(self)
        dialog.title("基础功率：输入目标转速")
        variables = {}
        for row, (key, label, value) in enumerate((("target_speed_rpm", "目标转速 r/min", "6000"),
                                                  ("initial_speed_rpm", "初始稳定转速 r/min", "0"))):
            ttk.Label(dialog, text=label).grid(row=row, column=0, padx=12, pady=8)
            variables[key] = tk.StringVar(value=value)
            ttk.Entry(dialog, textvariable=variables[key], width=16).grid(row=row, column=1, padx=12, pady=8)
        ttk.Label(dialog, text="启动时初始填0；制动时目标填0。\n时间从观测响应起点计算。仅限已覆盖的升降速设置。").grid(row=2, column=0, columnspan=2, padx=12, pady=8)
        def submit():
            try:
                accept({key: float(value.get()) for key, value in variables.items()})
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("转速查询", str(exc), parent=dialog)
        ttk.Button(dialog, text="生成曲线和参数", command=submit).grid(row=3, column=0, columnspan=2, pady=10)

    def _subtract(self):
        if self.base is None:
            raise ValueError("先标定或加载基础模型")
        path = filedialog.askopenfilename(parent=self, filetypes=[("加工CSV", "*.csv")])
        if not path:
            return
        model = self.base
        def operation(cancel, reference, session):
            record = read_measurement(path, cancel=cancel)
            if reference is not None:
                verify_reference_record(record, reference)
            result = predict_base_from_measured_speed(record.time_s, record.values["spindle_speed_rpm"], model,
                                                      reference=reference, session=session, record_source=record.source)
            return record, result
        def done(result):
            record, base = result
            series = {"主轴功率": record.values["spindle_power_w"]}
            if "power_unadjusted_w" in base:
                series["校正前基础功率"] = base["power_unadjusted_w"]
            series.update({"基础功率": base["power_w"], "切削功率": record.values["spindle_power_w"] - base["power_w"]})
            self.plot(record.time_s, series)
            self.show({"base_status": base["status"], "warnings": base["warnings"],
                       "空载参考": base.get("reference_adjustment"), "有效点数": int(base["valid"].sum())})
        self._with_idle_reference(model, path, lambda reference, session:
                                  self.start("计算实测切削功率…", lambda c: operation(c, reference, session), done))

    def _forecast_base(self):
        if self.base is None or self.feed_plan is None:
            raise ValueError("需要基础模型和指令进给计划")
        if not self.feed_plan.timing_complete:
            raise ValueError("主轴等待/人工暂停时序尚未配置，不能生成整程序绝对时间预测")
        model, plan = self.base, self.feed_plan
        source = "NC:" + plan.program_version
        def operation(cancel, reference, session):
            end = plan.report["total_automatic_time_s"]
            time = np.arange(0, end + .001, .001)
            commands = np.zeros(len(time))
            for event in plan.report["events"]:
                commands[time >= event["time_s"]] = event["spindle_rpm"]
            return predict_base_from_commands(time, commands, model, source="nc_aligned", reference=reference,
                                              session=session, record_source=source, cancel=cancel)
        def done(result):
            series = {"预测基础功率": result["power_w"]}
            if "power_unadjusted_w" in result:
                series["校正前基础功率"] = result["power_unadjusted_w"]
            self.plot(result["time_s"], series)
            self.show({k: result.get(k) for k in ("status", "timing", "warnings", "reference_adjustment", "prediction_version")})
        self._with_idle_reference(model, source, lambda reference, session:
                                  self.start("预测加工前基础功率…", lambda c: operation(c, reference, session), done))

    def _edit_machine(self):
        dialog = tk.Toplevel(self)
        dialog.title("机床运动参数（留空表示待配置）")
        dialog.geometry("780x670")
        fields = [("name", "机床名称"), ("max_velocity_mm_min", "XYZ最大速度 mm/min"),
                  ("max_acceleration_mm_s2", "XYZ最大加速度 mm/s²"), ("max_jerk_mm_s3", "XYZ最大跃度 mm/s³"),
                  ("interpolation_period_s", "插补周期 s"), ("contour_tolerance_mm", "轮廓容差 mm"),
                  ("corner_tolerance_mm", "拐角容差 mm"), ("initial_position_mm", "初始XYZ mm"),
                  ("travel_min_mm", "XYZ行程下限 mm"), ("travel_max_mm", "XYZ行程上限 mm"),
                  ("home_position_mm", "参考点XYZ mm"), ("spindle_wait_s", "主轴等待 s"),
                  ("tool_change_s", "换刀等待 s"), ("provenance", "参数来源（测试参数填 test）")]
        values = {}
        lists = {k for k, _ in fields if isinstance(getattr(self.machine, k), list) or k in ("max_velocity_mm_min", "max_acceleration_mm_s2", "max_jerk_mm_s3", "initial_position_mm", "travel_min_mm", "travel_max_mm", "home_position_mm")}
        for i, (key, label) in enumerate(fields):
            ttk.Label(dialog, text=label).grid(row=i, column=0, sticky="w", padx=8, pady=3)
            current = getattr(self.machine, key)
            value = tk.StringVar(value="" if current is None else ", ".join(map(str, current)) if isinstance(current, list) else str(current))
            ttk.Entry(dialog, textvariable=value, width=58).grid(row=i, column=1, padx=8)
            values[key] = value
        ttk.Label(dialog, text="XYZ各填三个数，以逗号分隔；坐标偏置、刀长和厂家语义在高级设置填写。").grid(row=len(fields), column=0, columnspan=2, padx=8, pady=8)
        def save(close=True):
            try:
                data = asdict(self.machine)
                for key, value in values.items():
                    raw = value.get().strip()
                    data[key] = raw if key in ("name", "provenance") else ([float(v) for v in raw.replace("，", ",").split(",")] if key in lists else float(raw)) if raw else None
                candidate = MachineConfig(**data)
                if asdict(candidate) != asdict(self.machine):
                    self._clear_idle_reference()
                self.machine = candidate
                self.show({"configuration": data, "missing": candidate.missing()})
                if close:
                    dialog.destroy()
                return True
            except Exception as exc:
                messagebox.showerror("参数格式", str(exc), parent=dialog)
                return False
        def advanced():
            if save():
                self._json_editor("完整机床配置：未核验厂家语义保持空白", asdict(self.machine), lambda v: setattr(self, "machine", MachineConfig(**v)))
        ttk.Button(dialog, text="保存参数", command=save).grid(row=len(fields) + 1, column=0, pady=8)
        ttk.Button(dialog, text="高级设置", command=advanced).grid(row=len(fields) + 1, column=1, pady=8)

    def _load_machine(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("机床JSON", "*.json")])
        if path:
            self.machine = MachineConfig.load(path)
            self._clear_idle_reference()
            self.show({"configuration": asdict(self.machine), "missing": self.machine.missing()})

    def _save_machine(self):
        path = self._choose_save("保存机床配置")
        if path:
            self.machine.save(path)

    def _parse_nc(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("NC程序", "*.nc *.NC *.tap *.txt"), ("全部文件", "*.*")])
        if not path:
            return
        self.nc_path = path
        config = self.machine
        def done(parsed):
            self.show({"program_version": parsed["program_version"], "segments": len(parsed["segments"]), "issues": parsed["issues"], "events": parsed["events"]})
            if parsed["segments"]:
                points = np.vstack([s.points(np.linspace(0, 1, 10)) for s in parsed["segments"]])
                label = "部分几何（配置缺失）" if parsed["issues"] else "数控程序几何"
                self.plot(points[:, 0], {label: points[:, 1]}, "X轴坐标（mm）", "Y轴坐标（mm）")
        self.start("解析NC…", lambda c: parse_nc(path, config), done)

    def _plan(self):
        if not self.nc_path:
            raise ValueError("先选择NC程序")
        path, config = self.nc_path, self.machine
        def done(result):
            self.feed_plan = result
            self.samples = self.forward_features = self.last_prediction = None
            self.show(result.report)
            self.plot(result.frame.time_s, {"规划进给": result.frame.feed_cmd_mm_min}, ylabel="进给速度（mm/min）")
            self.status.set("进给估计已生成；实际控制器一致性待验证")
        self.start("路径平滑与速度规划…", lambda c: plan_feed(path, config, cancel=c), done)

    def _save_plan(self):
        if self.feed_plan is None:
            raise ValueError("尚无运动计划")
        path = self._choose_save("保存运动计划")
        if path:
            self.feed_plan.save(path)

    def _load_plan(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("计划JSON", "*.json")])
        if path:
            self.feed_plan = FeedPlan.load(path)
            self.samples = self.forward_features = self.last_prediction = None
            self.show(self.feed_plan.report)

    def _edit_model(self):
        def accept(value):
            config = ModelConfig(**value)
            config.validate()
            self.model_config = config
        self._json_editor("候选网络配置；最终规模由后续9折验证决定", asdict(self.model_config), accept)

    def _build_models(self):
        config = self.model_config
        def done(bundle):
            self.bundle = bundle
            self.show({"status": "结构构建成功，未训练", "main_parameters": sum(p.numel() for p in bundle.main.parameters()),
                       "residual_parameters": sum(p.numel() for p in bundle.residual.parameters()), "config": asdict(config)})
        self.start("检查神经模型依赖与结构…", lambda c: build_load_models(config), done)

    def _prepare(self):
        if self.base is None or self.feed_plan is None:
            raise ValueError("先加载基础模型和对应程序的进给计划")
        process_path = filedialog.askopenfilename(parent=self, title="工艺表：标准CSV或当前工艺TXT", filetypes=[("工艺表", "*.csv *.txt")])
        if not process_path:
            return
        sample_path = filedialog.askopenfilename(parent=self, title="对应轮次实测CSV", filetypes=[("CSV", "*.csv")])
        if not sample_path:
            return
        metadata = dict(round_id=None, combination=None, parameter_group=None, feature=None, tool_session=None,
                        diameter_mm=None, teeth=None, program_version=self.feed_plan.program_version,
                        tool_healthy=None, no_alarm=None, complete=None, alignment_reliable=None,
                        process_line_offset=0, measurement_line_offset=0, steady_lines=[])
        plan, base = self.feed_plan, self.base
        def launch(reference, session):
            def accept(info):
                def operation(cancel):
                    details = dict(info)
                    process = read_process_table(process_path, plan, line_offset=details.pop("process_line_offset", 0), steady_lines=details.pop("steady_lines", []))
                    record = read_measurement(sample_path, cancel=cancel)
                    offset = details.pop("measurement_line_offset", 0)
                    raw_lines = record.values["program_line_raw"]
                    mapping = {int(n): int(n) + int(offset) for n in np.unique(raw_lines[np.isfinite(raw_lines)])}
                    return prepare_load_samples(process, plan, record, base, details, line_map=mapping,
                                                reference=reference, session=session)
                def done(result):
                    self.samples = result
                    self.forward_features = None
                    self.show({"metadata": result.metadata, "training_manifest": train_two_stage([result]),
                               "candidate_points": len(result.frame), "aligned_points": int(result.frame.aligned.sum())})
                self.start("对齐工艺与实测，准备候选样本…", operation, done)
            self._json_editor("单轮元数据；未知保持null，稳态行必须经过审核", metadata, accept)
        self._with_idle_reference(base, sample_path, launch)

    def _export_samples(self):
        if self.samples is None:
            raise ValueError("先准备候选样本")
        path = self._choose_save("导出候选样本", ".csv")
        if path:
            self.samples.frame.to_csv(path, index=False, encoding="utf-8")
            save_json(str(path) + ".metadata.json", self.samples.metadata)

    def _save_models(self):
        if self.bundle is None:
            raise ValueError("尚未构建模型")
        path = self._choose_save("保存模型包（未训练状态会保留）", ".pt")
        if path:
            self.bundle.save(path)

    def _load_models(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("PyTorch模型", "*.pt")])
        if path:
            def done(bundle):
                self.bundle = bundle
                self.show({"main_trained": bundle.main_trained, "residual_trained": bundle.residual_trained, "metadata": bundle.metadata})
            self.start("加载模型…", lambda c: load_models(path), done)

    def _predict_load(self):
        source = self.forward_features if self.forward_features is not None else self.samples
        if self.bundle is None or source is None:
            raise ValueError("需要已训练模型包和已准备的工艺特征")
        self.bundle.assert_ready()
        frame = source.frame.copy()
        valid = np.isfinite(frame[PHYSICS_COLUMNS]).all(axis=1).to_numpy()
        frame["sequence_id"] = np.cumsum(np.r_[True, ~valid[1:] | ~valid[:-1] | (np.diff(frame.execution_instance) != 0)])
        frame = frame.loc[valid].reset_index(drop=True)
        if frame.empty:
            raise ValueError("没有完整的加工前特征")
        bundle, metadata = self.bundle, dict(source.metadata)
        memory = self.memory if all(metadata.get(k) for k in ("round_id", "program_version", "parameter_group", "tool_session")) else None
        def operation(cancel):
            checkpoint(cancel)
            return predict_cutting_power(bundle, frame, memory=memory, context=metadata, round_id=metadata.get("round_id"))
        def done(result):
            self.last_prediction = (frame, result, metadata, bundle.main_version, bundle.residual_version)
            self.plot(frame.path_mm, {"主趋势": result["main_w"], "路径残差": result["residual_w"], "同位置记忆": result["memory_w"], "切削功率": result["cutting_power_w"]}, "路径行程（mm）")
            self.show({"main_version": bundle.main_version, "residual_version": bundle.residual_version, "points": len(frame)})
        self.start("预测功率分解…", operation, done)

    def _prepare_forward(self):
        if self.feed_plan is None:
            raise ValueError("先生成或加载进给计划")
        path = filedialog.askopenfilename(parent=self, title="选择加工前工艺表", filetypes=[("工艺表", "*.csv *.txt")])
        if not path:
            return
        plan = self.feed_plan
        metadata = dict(diameter_mm=None, teeth=None, program_version=plan.program_version,
                        round_id=None, parameter_group=None, tool_session=None, process_line_offset=0)
        def accept(info):
            def operation(cancel):
                process = read_process_table(path, plan, line_offset=info.pop("process_line_offset", 0))
                checkpoint(cancel)
                return prepare_forward_features(process, plan, info)
            def done(result):
                self.forward_features = result
                self.show({"metadata": result.metadata, "points": len(result.frame), "input": "工艺表＋NC计划，无实测输入"})
            self.start("准备加工前特征…", operation, done)
        self._json_editor("刀具信息必填；无历史记忆时轮次/会话可为空", metadata, accept)

    def _update_memory(self):
        if self.last_prediction is None:
            raise ValueError("先完成该轮的加工前预测，再使用该轮实测误差更新")
        frame, result, metadata, main_version, residual_version = self.last_prediction
        if "power_cut_w" not in frame or not np.isfinite(frame.power_cut_w).all():
            raise ValueError("该轮实测切削功率不完整，暂停记忆更新")
        error = frame.power_cut_w.to_numpy() - result["main_w"] - result["residual_w"]
        context = {**metadata, "main_version": main_version, "residual_version": residual_version}
        self.show(self.memory.update_round(context, frame, error, metadata["round_id"], healthy=metadata.get("tool_healthy"),
                    no_alarm=metadata.get("no_alarm"), complete=metadata.get("complete"), aligned=metadata.get("alignment_reliable")))

    def _memory_dialog(self):
        dialog = tk.Toplevel(self)
        dialog.title("同位置记忆")
        def save():
            path = self._choose_save("保存记忆")
            if path:
                self.memory.save(path)
                dialog.destroy()
        def load():
            path = filedialog.askopenfilename(parent=dialog, filetypes=[("记忆JSON", "*.json")])
            if path:
                try:
                    self.memory = PositionMemory.load(path)
                    self.last_prediction = None
                    dialog.destroy()
                except Exception as exc:
                    messagebox.showerror("记忆加载", str(exc), parent=dialog)
        ttk.Button(dialog, text="保存", command=save).pack(side=tk.LEFT, padx=12, pady=12)
        ttk.Button(dialog, text="加载", command=load).pack(side=tk.LEFT, padx=12, pady=12)

    def _reset_memory(self):
        self.memory = PositionMemory()
        self.last_prediction = None
        self.status.set("记忆已重置；下一轮为冷启动")
