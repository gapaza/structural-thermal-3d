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


















if __name__ == '__main__':

    nelx = 4
    nely = 4

    node_TL = 0
    node_BL = nely
    node_TR = nelx * (nely + 1)
    node_BR = nely + ((nely + 1) * nelx)

    nodes_L = np.arange(node_TL, node_BL + 1)
    nodes_R = np.arange(node_TR, node_BR + 1)
    nodes_T = np.arange(node_TL, node_TR + 1, nely+1)
    nodes_B = np.arange(node_BL, node_BR + 1, nely+1)


    print('Node TL', node_TL)
    print('Node BL', node_BL)
    print('Node TR', node_TR)
    print('Node BR', node_BR)

    print('Nodes L', nodes_L)
    print('Nodes R', nodes_R)
    print('Nodes T', nodes_T)
    print('Nodes B', nodes_B)