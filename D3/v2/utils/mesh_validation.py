import trimesh
import numpy as np

# Load the mesh you just saved
mesh_path = '/D3/v2/designs/voxel_mesh3.stl'
mesh = trimesh.load(mesh_path)

print("--- Mesh Diagnostics ---")
print(f"Is Watertight?      {mesh.is_watertight}")
print(f"Is Winding Correct? {mesh.is_winding_consistent}")
print(f"Euler Number:       {mesh.euler_number}")
print(f"Volume:             {mesh.volume:.4f}")

# Detailed check for issues
if not mesh.is_watertight:
    # Check for broken edges (edges shared by only 1 face instead of 2)
    broken_edges = len(mesh.edges_unique) - len(mesh.edges_unique_inverse)
    print(f"Found {broken_edges} broken (open) edges.")

    # Check for non-manifold vertices (vertices that pinch the mesh)
    # This is the "kissing corner" problem common in voxel data
    print(f"Non-manifold vertices: {len(mesh.process.find_mergeable())}")

else:
    print("SUCCESS: Mesh is solid and ready for 3D printing.")