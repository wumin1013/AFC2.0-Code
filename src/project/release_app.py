from __future__ import annotations

from .academic_workbench import AcademicWorkbenchMixin
from .analysis_export import AnalysisExportMixin
from .config_state import ConfigStateMixin
from .input_idle import InputIdleMixin
from .interval_runtime import IntervalRuntimeMixin
from .plot_support import PlotSupportMixin
from .processing_core import ProcessingCoreMixin
from .sample_manager import SampleManagerMixin
from .ui_bootstrap import BootstrapUiMixin


class AFCReleaseApplication(
    BootstrapUiMixin,
    AcademicWorkbenchMixin,
    InputIdleMixin,
    IntervalRuntimeMixin,
    ProcessingCoreMixin,
    SampleManagerMixin,
    PlotSupportMixin,
    AnalysisExportMixin,
    ConfigStateMixin,
):
    """AFC2.0.2alpha 多文件发布版主类（不含 PIT/SMIF 混入层）。"""

    release_mode = True
    enable_research_features = False
    enable_profile_config = False

# 供启动器和外部冒烟测试使用的稳定名称。
MillingAnalysisTool = AFCReleaseApplication

__all__ = ["AFCReleaseApplication", "MillingAnalysisTool"]
