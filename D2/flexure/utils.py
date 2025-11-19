"""Utility functions for the flexure problem."""

from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np





def plot_design(design, open_plot=True):
    # Create Plots
    fig, ax = plt.subplots(1, 2, figsize=(7, 5))

    ax[0].imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax[0].axis("off")
    ax[0].set_title("Design")

    plt.tight_layout()
    if open_plot is True:
        plt.show()
    return fig