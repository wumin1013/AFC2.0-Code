from __future__ import annotations

from .academic_workbench import AcademicWorkbenchMixin
from .analysis_export import AnalysisExportMixin
from .config_state import ConfigStateMixin
from .input_idle import InputIdleMixin
from .pit_model import PitModelMixin
from .plot_support import PlotSupportMixin
from .processing_core import ProcessingCoreMixin
from .sample_manager import SampleManagerMixin
from .ui_bootstrap import BootstrapUiMixin


class MillingAnalysisTool(
    BootstrapUiMixin,
    AcademicWorkbenchMixin,
    InputIdleMixin,
    PitModelMixin,
    ProcessingCoreMixin,
    SampleManagerMixin,
    PlotSupportMixin,
    AnalysisExportMixin,
    ConfigStateMixin,
):
    """模块化后的铣削工艺信息分析工具主类。"""

    pass
