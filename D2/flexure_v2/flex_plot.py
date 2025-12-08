import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors


def plot_thermal_actuation(optimizer, U_disp, Phi_final, title="Thermal Actuation"):
    """
    Visualizes Material Layout, Temperature Field, and Thermally Deformed Shape.
    Args:
        optimizer: Instance of ThermalFlexureTopologyOptimization
        U_disp: Nodal displacement vector (U1)
        Phi_final: Nodal temperature vector at final time tf
        title: Plot title
    """
    # plt.ion()  # Enable interactive mode

    # # 1. Robust figure initialization
    # if not hasattr(plot_thermal_actuation, "fig") or \
    #         not plt.fignum_exists(plot_thermal_actuation.fig.number):
    #     # Create a 1x3 grid, make it wider
    #     plot_thermal_actuation.fig, plot_thermal_actuation.ax = plt.subplots(1, 3, figsize=(16, 5))

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    ax1, ax2, ax3 = ax[0]
    ax4, ax5, ax6 = ax[1]

    # Clear previous frame
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # # --- Shared Data ---
    # # Node grid coordinates
    X_nodes, Y_nodes = np.meshgrid(np.arange(optimizer.nelx + 1), np.arange(optimizer.nely + 1))
    #
    # # --- Plot 1: Material Layout ---
    # # Reshape x to (nelx, nely) then Transpose for image coords
    # x_grid = optimizer.x.reshape(optimizer.nelx, optimizer.nely).T
    #
    # ax1.imshow(1 - x_grid, cmap='gray',
    #            extent=[0, optimizer.nelx, 0, optimizer.nely])
    #
    # ax1.set_title(f"1. Design (Vol: {np.mean(optimizer.x):.2f})")
    # ax1.set_xlabel("x")
    # ax1.set_ylabel("y")
    # ax1.set_aspect('equal')

    design = optimizer.x.reshape(optimizer.nelx, optimizer.nely).T
    ax1.imshow(-design, cmap="gray", interpolation="none", norm=colors.Normalize(vmin=-1, vmax=0))
    ax1.axis("off")
    ax1.set_title("Design")



    # --- Plot 2: Temperature Distribution ---
    # Reshape nodal temperature vector. Nodes are column-major: n = ely + elx * (nely + 1)
    # Reshape to (nelx+1, nely+1) then transpose to (nely+1, nelx+1) for plotting
    phi_grid = Phi_final.reshape((optimizer.nelx + 1, optimizer.nely + 1)).T

    # # Plot using pcolormesh on the node grid
    # # cmap='plasma' or 'inferno' are good for heat
    # im2 = ax2.pcolormesh(X_nodes, Y_nodes, phi_grid, cmap='plasma', shading='gouraud')
    #
    # ax2.set_title("2. Temperature (T)")
    # ax2.set_xlabel("x")
    # ax2.set_aspect('equal')
    #
    # # Add colorbar for temperature
    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    # plt.colorbar(im2, cax=cax2, label='Temp')

    t_xxx = np.arange(optimizer.nelx + 1)
    t_yyy = -np.arange(optimizer.nely + 1)
    ax2.contourf(t_xxx, t_yyy, phi_grid, 50)
    ax2.axis("image")
    ax2.set_title("Temperature")


    # --- Plot 3: Deformed Shape ---
    # Extract U and V displacements (Column-Major handling)
    u_vals = U_disp[0::2].reshape((optimizer.nelx + 1, optimizer.nely + 1)).T
    v_vals = U_disp[1::2].reshape((optimizer.nelx + 1, optimizer.nely + 1)).T
    mag = np.sqrt(u_vals ** 2 + v_vals ** 2)

    # Auto-scale deformation
    max_disp = np.max(mag)
    scale_factor = 1.0
    if max_disp > 0:
        # Scale max deformation to 15% of the domain size
        scale_factor = 0.15 * max(optimizer.nelx, optimizer.nely) / max_disp

    X_def = X_nodes + u_vals * scale_factor
    Y_def = Y_nodes + v_vals * scale_factor

    # Plot Deformed Mesh
    im3 = ax3.pcolormesh(X_def, Y_def, mag, cmap='turbo', shading='gouraud')

    # Add boundary contour for clarity
    ax3.contour(X_def, Y_def, mag, colors='k', linewidths=0.5, alpha=0.3)

    ax3.set_title(f"3. Deformation (scaled {scale_factor:.1f}x)")
    ax3.set_xlabel("x")
    ax3.set_aspect('equal')

    # Set limits with a small buffer
    buff = 0.1 * max(optimizer.nelx, optimizer.nely)
    ax3.set_xlim(-buff, optimizer.nelx + buff)
    ax3.set_ylim(-buff, optimizer.nely + buff)
    ax3.invert_yaxis()

    # Add colorbar for displacement magnitude
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im3, cax=cax3, label='Disp. Mag.')

    # --- Plot 4: x displacements ---

    im4 = ax4.imshow(u_vals)
    ax4.set_title("x-displacements")
    divider4 = make_axes_locatable(ax4)
    cax4 = divider4.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im4, cax=cax4)

    # --- Plot 5: y displacements ---

    im5 = ax5.imshow(v_vals)
    ax5.set_title("y-displacements")
    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im5, cax=cax5)

    plt.suptitle(title)

    # Force update
    # plot_thermal_actuation.fig.canvas.draw()
    # plot_thermal_actuation.fig.canvas.flush_events()
    # Pause needed for GUI to catch up
    # plt.pause(0.1)
    plt.show()
    plt.close()



def plot_temperatures(array_list, time_list, opt, suptitle=None):
    num_subplots = 25
    selected_indices = np.linspace(0, len(array_list) - 1, num_subplots, dtype=int)
    fig, axes = plt.subplots(5, 5, figsize=(15, 15), constrained_layout=True)
    flat_axes = axes.flatten()
    for i, ax in enumerate(flat_axes):
        # Get the specific index from our sampled list
        idx = selected_indices[i]
        data = array_list[idx]
        data = np.reshape(data, (opt.nelx+1, opt.nely+1)).T
        timestamp = time_list[idx]
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f"t = {timestamp:.2f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')
    if suptitle:
        plt.suptitle(suptitle)
    plt.show()










