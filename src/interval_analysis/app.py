from __future__ import annotations

from .academic_workbench import AcademicWorkbenchMixin
from .analysis_export import AnalysisExportMixin
from .config_state import ConfigStateMixin
from .input_idle import InputIdleMixin
from .interval_runtime import IntervalRuntimeMixin
from .pit_viewer import PitViewerMixin
from .pit_model import MechanismModelMixin
from .plot_support import PlotSupportMixin
from .prediction_runtime import PredictionRuntimeMixin
from .processing_core import ProcessingCoreMixin
from .sample_manager import SampleManagerMixin
from .ui_bootstrap import BootstrapUiMixin


class MillingAnalysisTool(
    BootstrapUiMixin,
    PitViewerMixin,
    PredictionRuntimeMixin,
    AcademicWorkbenchMixin,
    InputIdleMixin,
    MechanismModelMixin,
    IntervalRuntimeMixin,
    ProcessingCoreMixin,
    SampleManagerMixin,
    PlotSupportMixin,
    AnalysisExportMixin,
    ConfigStateMixin,
):
    """模块化后的铣削工艺信息分析工具主类。"""

    pass
