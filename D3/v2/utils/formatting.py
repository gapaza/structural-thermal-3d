import numpy as np
import napari




numpy_file = '/D3/v2/designs/design.npz'

# Load numpy data

data = np.load(numpy_file)


design = data['design']
fixed_therm_nodes = data['fixed_therm_nodes']
heat_gen_nodes = data['heat_gen_nodes']
fixed_el_nodes = data['fixed_el_nodes']
loaded_el_nodes = data['loaded_el_nodes']
print('Design shape:', design.shape)



threshold = 0.5
design = (design >= threshold).astype(int)




viewer = napari.Viewer()
viewer.add_image(design, name='rho', rendering='attenuated_mip')
viewer.add_image(fixed_el_nodes, name='fixed_elements', rendering='attenuated_mip', visible=False, colormap='green')
viewer.add_image(loaded_el_nodes, name='force_elements', rendering='attenuated_mip', visible=False, colormap='fire')
viewer.add_image(fixed_therm_nodes, name='heatsink_elements', rendering='attenuated_mip', visible=False, colormap='purple')
viewer.add_image(heat_gen_nodes, name='heat_gen_elements', rendering='attenuated_mip', visible=False, colormap='blue')
viewer.dims.ndisplay = 3  # switch to 3D view
napari.run()



# New save file
new_save = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/designs/voxel_design.npz'
np.savez(
    new_save,
    design=design,
)

