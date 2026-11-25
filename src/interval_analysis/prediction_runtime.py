from __future__ import annotations

from .shared import *


class PredictionRuntimeMixin:
    """主页面预测运行时。

    PIT 仅消费工艺信息；实验实测反解、profile 前向预测和预测通道更新
    都由本运行时统一编排。
    """

    def has_prediction_model_ready(self, process_path=None):
        if self.has_identified_kc_ke():
            return True
        profile = self._get_saved_kc_profile_for_input(process_path)
        return self._profile_has_saved_payload(profile)

    def _apply_current_interval_mode_kc_override_to_measurement(self):
        if getattr(self, "sample_data_mode", "") != "experiment_measurement":
            return None
        if not getattr(self, "manual_measurement_data", None):
            return None
        if self._is_imported_profile_forward_lock_active():
            return None
        if not self._get_current_interval_records(allow_profile_fallback=False):
            return None
        return self.refresh_main_prediction(
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
        """兼容既有调用点，实际预测统一进入主页面预测运行时。"""
        return self.refresh_main_prediction(
            allow_saved_sample_profile=allow_saved_sample_profile,
            allow_measurement_resolve=allow_measurement_resolve,
            display_mode=display_mode,
        )

    def refresh_main_prediction(
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
            "refresh_main_prediction",
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
