"""Shared matplotlib settings for the JAX-GalSim paper figures."""

import matplotlib as mpl

# openjournal.cls (two-column, 10pt)
TEXT_WIDTH = 7.1  # inches, full page width (figure*)
COLUMN_WIDTH = (7.1 - 0.3125) / 2  # inches, one column

DPI = 300

# residual colormap (diverging, neutral midpoint) and image colormap (sequential)
RES_CMAP = "RdBu_r"
IMAGE_CMAP = "gray_r"


def set_style():
    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            # LaTeX for all text, with the paper's font (T1 Computer Modern, see macros.sty)
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def pixel_axes(axes, ticks):
    """Same pixel ticks on a row of image panels.

    Every panel gets tick marks and x tick labels; only the first panel gets y tick
    labels.
    """
    for i, ax in enumerate(axes):
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.tick_params(direction="in", length=2.5, color="0.5", top=True, right=True)
        if i > 0:
            ax.tick_params(labelleft=False)
