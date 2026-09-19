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

# Directory for saved figures, resolved relative to the repo root
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Figs")


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
