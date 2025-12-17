import numpy as np
from skimage import measure, morphology
from stl import mesh
import os



# --- 1. Load the 3D NumPy Array (Voxel Grid) ---

numpy_file = '/D3/v2/designs/voxel_design.npz'
data = np.load(numpy_file)
voxel_array = data['design']


voxel_array = np.transpose(voxel_array, (2, 1, 0))





# --- 2. Mesh the design ---
padded_array = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=0)

# New shape will be (18, 42, 102)
print(f"Old shape: {voxel_array.shape}")
print(f"New shape: {padded_array.shape}")

# Mesh the PADDED array
verts, faces, normals, values = measure.marching_cubes(padded_array, level=0.5)

# Save (Standard procedure)
mesh_data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
for i, f in enumerate(faces):
    mesh_data['vectors'][i] = verts[f]

mesh_object = mesh.Mesh(mesh_data)
# save_path = os.path.join('/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/designs', 'voxel_mesh3.stl')
# mesh_object.save(save_path)



# --- 3. (Optional) Basic Visualization ---
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
mesh_plot = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral', lw=1)

# --- CRITICAL FIX: Set Equal Aspect Ratio ---

# 1. Find the maximum range of your data to center the view
max_range = np.array([verts[:, 0].max()-verts[:, 0].min(),
                      verts[:, 1].max()-verts[:, 1].min(),
                      verts[:, 2].max()-verts[:, 2].min()]).max() / 2.0

mid_x = (verts[:, 0].max()+verts[:, 0].min()) * 0.5
mid_y = (verts[:, 1].max()+verts[:, 1].min()) * 0.5
mid_z = (verts[:, 2].max()+verts[:, 2].min()) * 0.5

# 2. Set the limits artificially to form a cube around your long object
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.invert_zaxis()


# --- NEW CODE: Adjust the viewing angle ---
# Example: 30 degrees elevation (from the horizon) and -60 degrees azimuth (horizontal rotation)
# ax.view_init(elev=10, azim=-30)
ax.view_init(elev=5, azim=0)
# ax.view_init(elev=90, azim=0)
# ---------------------------------


plt.show()



