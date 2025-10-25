from __future__ import annotations

from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

# import plotly.graph_objects as go
# from skimage.measure import marching_cubes


import napari



def plot_fem_3d(bcs, design):
    nelx, nely, nelz = design.shape

    fixed_elements = bcs.get("fixed_elements", np.zeros_like(design))

    force_elements_x = bcs.get("force_elements_x", np.zeros_like(design))
    force_elements_y = bcs.get("force_elements_y", np.zeros_like(design))
    force_elements_z = bcs.get("force_elements_z", np.zeros_like(design))
    force_elements = force_elements_x + force_elements_y + force_elements_z

    heatsink_elements = bcs.get("heatsink_elements", np.zeros_like(design))


    design      = np.transpose(design, (2, 0, 1))
    fixed_elements = np.transpose(fixed_elements, (2, 1, 0))
    force_elements = np.transpose(force_elements, (2, 1, 0))
    heatsink_elements = np.transpose(heatsink_elements, (2, 1, 0))


    viewer = napari.Viewer()
    viewer.add_image(design, name='rho', rendering='attenuated_mip')
    viewer.add_image(fixed_elements, name='fixed_elements', rendering='attenuated_mip', visible=False, colormap='green')
    viewer.add_image(force_elements, name='force_elements', rendering='attenuated_mip', visible=False, colormap='fire')
    viewer.add_image(heatsink_elements, name='heatsink_elements', rendering='attenuated_mip', visible=False, colormap='purple')


    viewer.dims.ndisplay = 3  # switch to 3D view
    # viewer.export_figure(path='/Users/gapaza/repos/ideal/structural-thermal-3d/figures/napari_fem_3d.png')
    napari.run()


