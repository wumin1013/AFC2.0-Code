"""Figures shared by embedded previews and offline exports."""
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def create_plot_figure(**kwargs):
    """Provide a renderer without a pyplot manager or a second Tk root.

    FigureCanvasTkAgg replaces this canvas when the desktop embeds the figure.
    Before embedding, Agg supports layout measurement and file export.
    """
    figure = Figure(**kwargs)
    FigureCanvasAgg(figure)
    return figure
