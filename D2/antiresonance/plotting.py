import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_eigenmodes(opt, x, freqs, modes, scale_factor=10.0):
    """
    Visualizes the material distribution and the first few mode shapes.

    Args:
        opt: The optimization object (contains nelx, nely).
        x: Design variable array (element densities).
        freqs: Array of natural frequencies (Hz).
        modes: Matrix of mode shapes (ndof x num_modes).
        scale_factor: Visual scaling for vibration amplitude (default: 10).
    """

    # 1. Setup Grid and Data
    # ----------------------
    num_modes = len(freqs)
    nelx, nely = opt.nelx, opt.nely

    # Create undeformed nodal grid (X, Y)
    # Note: Meshgrid indexing='xy' puts x on columns, y on rows
    X_grid, Y_grid = np.meshgrid(np.arange(nelx + 1), np.arange(nely + 1))

    # Reshape density x for plotting (flip y to match matrix indexing image)
    # We use 1-x so high density appears dark (like ink)
    density_grid = 1 - x.reshape((nely, nelx)).T

    # 2. Plotting
    # -----------
    fig, axes = plt.subplots(1, num_modes + 1, figsize=(3 * (num_modes + 1), 4))

    # Plot A: Undeformed Topology
    ax_topo = axes[0]
    ax_topo.imshow(density_grid, cmap='gray', extent=[0, nelx, 0, nely])
    ax_topo.set_title("Design Topology")
    ax_topo.axis('off')

    # Plot B: Mode Shapes (Deformed Meshes)
    for i in range(num_modes):
        ax = axes[i + 1]
        mode_vec = modes[:, i]
        freq = freqs[i]

        # Extract U and V displacements
        # DOF map: [u1, v1, u2, v2, ...]
        u_disp = mode_vec[0::2].reshape((nely + 1, nelx + 1))
        v_disp = mode_vec[1::2].reshape((nely + 1, nelx + 1))

        # Apply deformation scale
        # Normalize mode for consistent visualization amplitude
        max_disp = np.max(np.abs(mode_vec))
        if max_disp > 0:
            scale = scale_factor * (nelx / 10.0) / max_disp  # Auto-scale relative to domain size
        else:
            scale = 0

        # Deform the grid (flip y for plotting to match imshow orientation)
        X_def = X_grid + scale * u_disp
        # Note: In matrix plotting, Y often goes down, but FEM Y goes up.
        # We handle this by plotting typically, but ensuring density matches.
        # Simple approach: Plot standard XY cartesian.
        Y_def = np.flipud(Y_grid) + scale * np.flipud(v_disp)

        # Plot density on deformed grid using pcolormesh
        # 'shading=flat' requires X/Y to be corners (size N+1) and C to be cells (size N)
        ax.pcolormesh(X_def, Y_def, density_grid, cmap='gray', shading='flat', edgecolors='none')

        ax.set_title(f"Mode {i + 1}: {freq:.6f} Hz")
        ax.axis('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()



def plot_design(opt, x):

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    design = x.reshape(opt.nelx, opt.nely).T
    ax.imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax.axis("off")
    ax.set_title("Design")

    plt.show()
    plt.close()


def plot_harmonic_sweep(opt, freqs_scan, frf, inv_frf):

    plt.figure(figsize=(10, 8))

    # Subplot 1: The Raw Physical Response (What an engineer looks at)
    plt.subplot(2, 1, 1)
    plt.plot(freqs_scan, frf, 'b.-')
    plt.yscale('log')  # Log scale is crucial for FRFs
    plt.title("Raw Frequency Response (Displacement)")
    plt.ylabel("Displacement Amplitude (Log)")
    plt.grid(True, which="both", ls="-")

    # Subplot 2: The Inverted Signal (What find_peaks looks at)
    plt.subplot(2, 1, 2)
    plt.plot(freqs_scan, inv_frf, 'r.-')
    plt.title("Inverted FRF (1/Disp) - Used for Peak Finding")
    plt.ylabel("Inverse Amplitude")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()












