import numpy as np
from skimage import measure, morphology
from stl import mesh
import os

# --- 1. Load the 3D NumPy Array ---
numpy_file = '/D3/v2/designs/voxel_design.npz'
data = np.load(numpy_file)
voxel_array = data['design']

# ==========================================
# --- 2. PRE-PROCESSING (CRITICAL FIX) ---
# ==========================================

# A. Keep only the Largest Connected Component
# This removes floating "dust" particles to ensure the print is a single part.
labels = measure.label(voxel_array, connectivity=1)  # connectivity=1 checks faces only
if labels.max() > 0:
    # Get the count of voxels in each label (0 is background)
    counts = np.bincount(labels.ravel())
    counts[0] = 0  # Ignore background
    largest_label = counts.argmax()

    # Create a mask of only the largest part
    voxel_array = (labels == largest_label)
    print(f"Isolated largest component. Dropped {counts.sum() - counts.max()} floating voxels.")
else:
    print("Warning: Array is empty!")

# B. Fix Non-Manifold Geometry (Closing)
# This fills in diagonal "cracks" and "kissing corners" (checkerboards)
# that create zero-thickness geometry which Slicers hate.
# A small radius (ball of 1 or 2) is usually sufficient.
voxel_array = morphology.binary_closing(voxel_array, morphology.ball(2))

# ==========================================

# --- 3. Mesh the design ---

# Pad to close boundaries (Your original correct step)
padded_array = np.pad(voxel_array, pad_width=1, mode='constant', constant_values=0)

print(f"Final meshing shape: {padded_array.shape}")

# Mesh the PADDED array
# Note: spacing=(1,1,1) ensures the STL is scaled to the voxel indices.
# You might want to change this to your actual physical dimensions (e.g., spacing=(0.5, 0.5, 0.5) mm).
verts, faces, normals, values = measure.marching_cubes(padded_array, level=0.5)

# Save
mesh_data = np.zeros(len(faces), dtype=mesh.Mesh.dtype)
for i, f in enumerate(faces):
    mesh_data['vectors'][i] = verts[f]

mesh_object = mesh.Mesh(mesh_data)
save_path = os.path.join('/D3/v2/designs', 'voxel_mesh3.stl')
mesh_object.save(save_path)
print(f"Saved solid mesh to: {save_path}")




# --- 4. (Optional) Basic Visualization ---
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

ax.set_xlabel("X (16)")
ax.set_ylabel("Y (40)")
ax.set_zlabel("Z (100)")

plt.show()






