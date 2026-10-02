# LLM-assisted (code level 4): structured and written by Claude (Opus 4.8,
# claude.ai, Sept-Oct 2026) from the author's weekly-exercise solutions.
# The author adapted, tested against the closed-form and library benchmarks,
# commented and verified it. See Appendix A of the report.
"""Shared constants and small plotting helpers for Project 1.

Colour constants use the Paul Tol qualitative palette, colour-blind safe.
All code comments are kept in ASCII only to avoid encoding issues.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Paul Tol colours, reused from the weekly notebooks
BLUE = "#004488"
RED = "#BB5566"
YELLOW = "#DDAA33"
GREY = "#777777"
GREEN = "#228833"

# One seed for the whole project, matching the weekly exercises
SEED = 2026

# Directory for saved figures. common.py lives in code/, so the repo root is
# one level up, and figures go into the results folder.
FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def savefig(name, fig=None, dpi=200):
    """Save a figure into the Figs directory as a PDF for the report.

    name is given without extension; a .pdf is appended.
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    path = os.path.join(FIG_DIR, name + ".pdf")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    return path
