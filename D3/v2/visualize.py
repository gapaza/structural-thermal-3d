
# -------------------------------------
# Loading the voxel design
# -------------------------------------

import numpy as np
numpy_file = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/designs/voxel_design.npz'
data = np.load(numpy_file)
design = data['design']
print(design.shape)



# -------------------------------------
# Plotting the voxel design with napari
# -------------------------------------

import napari
viewer = napari.Viewer()
viewer.add_image(design, name='rho', rendering='attenuated_mip')
viewer.dims.ndisplay = 3  # switch to 3D view
napari.run()