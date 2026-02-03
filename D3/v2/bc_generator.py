"""Generates a set of boundary conditions and saves them to a file."""
import numpy as np
import pickle
import os
import secrets
import string

""" Dataset BC Definitions

# --- Structural BCs ---

Number of Supports: 3, 4
Support Size: 2x2 patch, 3x3 patch
Support locations: bottom face, back face, left face
Number of Loads: 1
Load Direction: x-y-z (all 3 directions always)
Load Locations: top face
Load Size: 2x2 patch
Volume Fractions: 0.3, 0.31, ..., 0.5

# --- Thermal BCs ---

Number of HeatSinks: 2, 3
HeatSink Size: 2x2 patch, 3x3 patch
HeatSink Locations: bottom face, back face, left face
Volume Fractions: 0.3, 0.31, ..., 0.5

"""




class BcGenerator:

    def __init__(self, nelx, nely, nelz):
        self.nelx, self.nely, self.nelz = nelx, nely, nelz

        # Preliminary boundary node definitions
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        x_range = np.arange(nx)
        z_range = np.arange(nz)
        y_range = np.arange(ny)

        # Top Face Nodes
        self.node_TF_TL = 0
        self.node_TF_ML = self.nely // 2
        self.node_TF_BL = self.nely
        self.node_TF_CENTER = (ny * (self.nelx // 2)) + (self.nely // 2)
        self.node_TF_TR = (ny * self.nelx)
        self.node_TF_BR = (ny * self.nelx) + self.nely
        self.nodes_TF = np.arange(self.node_TF_TL, self.node_TF_BR + 1)

        # Middle Face Nodes
        self.node_MF_TL = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_TL
        self.node_MF_ML = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_ML
        self.node_MF_BL = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_BL
        self.node_MF_CENTER = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_CENTER
        self.node_MF_TR = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_TR
        self.node_MF_BR = (nodes_per_slice * (self.nelz // 2)) + self.node_TF_BR
        self.nodes_MF = np.arange(self.node_MF_TL, self.node_MF_BR + 1)

        # Bottom Face Nodes
        self.node_BF_TL = (nodes_per_slice * self.nelz) + self.node_TF_TL
        self.node_BF_ML = (nodes_per_slice * self.nelz) + self.node_TF_ML
        self.node_BF_BL = (nodes_per_slice * self.nelz) + self.node_TF_BL
        self.node_BF_CENTER = (nodes_per_slice * self.nelz) + self.node_TF_CENTER
        self.node_BF_TR = (nodes_per_slice * self.nelz) + self.node_TF_TR
        self.node_BF_BR = (nodes_per_slice * self.nelz) + self.node_TF_BR
        self.nodes_BF = np.arange(self.node_BF_TL, self.node_BF_BR + 1)

        # Left Face (x = 0)
        # Formula: Index = (z * nodes_per_slice) + (0 * ny) + y
        # We use broadcasting: z_range[:, None] is column vector, y_range[None, :] is row vector
        self.nodes_LF = (z_range[:, None] * nodes_per_slice + y_range[None, :]).flatten()

        # Right Face (x = nelx)
        # This is just the Left Face offset by the width of the domain (nelx * ny)
        self.nodes_RF = self.nodes_LF + (self.nelx * ny)

        # Back Face (y = nely)
        # Corresponds to the "Bottom Edge" of every z-slice (TF_BL to TF_BR)
        # This is the Front Face offset by the height of the slice (nely)
        self.nodes_BackF = (z_range[:, None] * nodes_per_slice + x_range[None, :] * ny).flatten()

        # Front Face (y = 0)
        # Corresponds to the "Top Edge" of every z-slice (TF_TL to TF_TR)
        # Formula: Index = (z * nodes_per_slice) + (x * ny) + 0
        self.nodes_FF = self.nodes_BackF + self.nely


        # Conditions storage
        self.volfrac_set = [round(x, 2) for x in np.arange(0.35, 0.51, 0.01)]
        self.volfrac_set_test = [round(x, 2) for x in np.arange(0.3, 0.35, 0.01)]

        self.support_locations = ['bottom_face', 'back_face', 'left_face']
        self.support_sizes = ['2x2', '3x3']
        self.load_locations = ['top_face']


        self.heatsink_locations = ['bottom_face', 'back_face', 'left_face']
        self.heatsink_sizes = ['2x2', '3x3']

        self.fixed_therm_temp = 0
        self.heat_gen_tref = 0.001

        self.mechanical_load = 1000.0


    def sample_elastic(self):
        n_supports = np.random.choice([3])
        support_size = np.random.choice(self.support_sizes)

        # Fixed Nodes
        fixed_el_nodes = np.array([])
        if support_size == '2x2':
            for x in range(n_supports):
                face = self.support_locations[x]
                # face = np.random.choice(self.support_locations)
                patch_nodes = self.sample_2x2(face)
                while np.any(np.isin(patch_nodes, fixed_el_nodes)):
                    patch_nodes = self.sample_2x2(face)
                fixed_el_nodes = np.concatenate((fixed_el_nodes, patch_nodes))
        elif support_size == '3x3':
            for x in range(n_supports):
                face = self.support_locations[x]
                # face = np.random.choice(self.support_locations)
                patch_nodes = self.sample_3x3(face)
                while np.any(np.isin(patch_nodes, fixed_el_nodes)):
                    patch_nodes = self.sample_3x3(face)
                fixed_el_nodes = np.concatenate((fixed_el_nodes, patch_nodes))
        else:
            raise ValueError(f"Unknown support size: {support_size}")

        fixed_el_nodes_dof_x = (3 * fixed_el_nodes) + 0
        fixed_el_nodes_dof_y = (3 * fixed_el_nodes) + 1
        fixed_el_nodes_dof_z = (3 * fixed_el_nodes) + 2
        fixed_el_dof = np.hstack([
            fixed_el_nodes_dof_x,
            fixed_el_nodes_dof_y,
            fixed_el_nodes_dof_z,
        ])


        # Load Nodes
        loaded_el_nodes = np.array([])
        face = self.load_locations[0]  # Always top face
        patch_nodes = self.sample_2x2(face)
        loaded_el_nodes = np.concatenate((loaded_el_nodes, patch_nodes))

        loaded_el_nodes_dof_x = (3 * loaded_el_nodes) + 0
        loaded_el_nodes_dof_y = (3 * loaded_el_nodes) + 1
        loaded_el_nodes_dof_z = (3 * loaded_el_nodes) + 2
        loaded_el_dof = np.hstack([
            loaded_el_nodes_dof_x,
            loaded_el_nodes_dof_y,
            loaded_el_nodes_dof_z,
        ])

        return {
            'mechanical_load': self.mechanical_load,
            'loaded_el_dof': loaded_el_dof.astype(int),
            'fixed_el_dof': fixed_el_dof.astype(int),

            'loaded_el_nodes': loaded_el_nodes.astype(int),
            'fixed_el_nodes': fixed_el_nodes.astype(int),
        }


    def sample_thermal(self):
        n_heatsinks = np.random.choice([2, 3])
        heatsink_size = np.random.choice(self.heatsink_sizes)

        # Fixed Temperature Nodes
        fixed_therm_nodes = np.array([])
        if heatsink_size == '2x2':
            for x in range(n_heatsinks):
                face = np.random.choice(self.heatsink_locations)
                patch_nodes = self.sample_2x2(face)
                while np.any(np.isin(patch_nodes, fixed_therm_nodes)):
                    patch_nodes = self.sample_2x2(face)
                fixed_therm_nodes = np.concatenate((fixed_therm_nodes, patch_nodes))
        elif heatsink_size == '3x3':
            for x in range(n_heatsinks):
                face = np.random.choice(self.heatsink_locations)
                patch_nodes = self.sample_3x3(face)
                while np.any(np.isin(patch_nodes, fixed_therm_nodes)):
                    patch_nodes = self.sample_3x3(face)
                fixed_therm_nodes = np.concatenate((fixed_therm_nodes, patch_nodes))
        else:
            raise ValueError(f"Unknown heatsink size: {heatsink_size}")


        # Heat Generation Nodes
        all_nodes = np.arange((self.nelx + 1) * (self.nely + 1) * (self.nelz + 1))
        heat_gen_nodes = np.setdiff1d(all_nodes, fixed_therm_nodes)


        return {
            'fixed_therm_nodes': fixed_therm_nodes.astype(int),
            'fixed_therm_temp': self.fixed_therm_temp,

            'heat_gen_nodes': heat_gen_nodes.astype(int),
            'heat_gen_tref': self.heat_gen_tref,
        }



    def generate(self, save_path, n_samples):
        all_bcs = []
        for x in range(n_samples):
            volfrac = np.random.choice(self.volfrac_set)

            elastic_bc = self.sample_elastic()
            thermal_bc = self.sample_thermal()

            combined_bc = {**elastic_bc, **thermal_bc}
            combined_bc['volfrac'] = volfrac
            all_bcs.append(combined_bc)

            salt_str = self.generate_salt()

            file_name = f'bc_sample_{salt_str}.pkl'
            file_path = os.path.join(save_path, file_name)

            records_name = f'bc_sample_{salt_str}_records.pkl'
            records_path = os.path.join(save_path, records_name)

            struct_name = f'bc_sample_{salt_str}_struct.pkl'
            struct_path = os.path.join(save_path, struct_name)

            therm_name = f'bc_sample_{salt_str}_therm.pkl'
            therm_path = os.path.join(save_path, therm_name)

            combined_bc['file_path'] = file_path
            combined_bc['records_path'] = records_path
            combined_bc['struct_path'] = struct_path
            combined_bc['therm_path'] = therm_path
            with open(file_path, 'wb') as f:
                pickle.dump(combined_bc, f)
            print(f'Saved BC sample {x+1}/{n_samples} to {file_path}')

        return all_bcs

    def generate_salt(self, length=8):
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))

    #   _____                       _ _               _    _      _
    #  / ____|                     | (_)             | |  | |    | |
    # | (___   __ _ _ __ ___  _ __ | |_ _ __   __ _  | |__| | ___| |_ __   ___ _ __ ___
    #  \___ \ / _` | '_ ` _ \| '_ \| | | '_ \ / _` | |  __  |/ _ \ | '_ \ / _ \ '__/ __|
    #  ____) | (_| | | | | | | |_) | | | | | | (_| | | |  | |  __/ | |_) |  __/ |  \__ \
    # |_____/ \__,_|_| |_| |_| .__/|_|_|_| |_|\__, | |_|  |_|\___|_| .__/ \___|_|  |___/
    #                        | |               __/ |               | |
    #                        |_|              |___/                |_|

    def sample_2x2(self, face):
        if face == 'left_face':
            return self.sample_random_left_face_2x2()
        elif face == 'right_face':
            return self.sample_random_right_face_2x2()
        elif face == 'back_face':
            return self.sample_random_back_face_2x2()
        elif face == 'front_face':
            return self.sample_random_front_face_2x2()
        elif face == 'top_face':
            return self.sample_random_top_face_2x2()
        elif face == 'bottom_face':
            return self.sample_random_bottom_face_2x2()
        else:
            raise ValueError(f"Unknown face: {face}")

    def sample_3x3(self, face):
        if face == 'left_face':
            return self.sample_random_left_face_3x3()
        elif face == 'right_face':
            return self.sample_random_right_face_3x3()
        elif face == 'back_face':
            return self.sample_random_back_face_3x3()
        elif face == 'front_face':
            return self.sample_random_front_face_3x3()
        elif face == 'top_face':
            return self.sample_random_top_face_3x3()
        elif face == 'bottom_face':
            return self.sample_random_bottom_face_3x3()
        else:
            raise ValueError(f"Unknown face: {face}")


    # ------------------------------------------
    # Left Face
    # ------------------------------------------

    def sample_random_left_face_2x2(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        start_y = np.random.randint(0, self.nely)
        start_z = np.random.randint(0, self.nelz)
        start_node_idx = (start_z * nodes_per_slice) + start_y
        patch_nodes = np.array([
            start_node_idx,  # Top-Left
            start_node_idx + 1,  # Top-Right
            start_node_idx + nodes_per_slice,  # Bottom-Left
            start_node_idx + nodes_per_slice + 1  # Bottom-Right
        ])
        return patch_nodes

    def sample_random_left_face_3x3(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        start_y = np.random.randint(0, self.nely-1)
        start_z = np.random.randint(0, self.nelz-1)
        start_node_idx = (start_z * nodes_per_slice) + start_y
        patch_nodes = np.array([
            start_node_idx,  # Top-Left
            start_node_idx + 1,  # Top-Middle
            start_node_idx + 2,  # Top-Right
            start_node_idx + nodes_per_slice,  # Center-Left
            start_node_idx + nodes_per_slice + 1,  # Center-Middle
            start_node_idx + nodes_per_slice + 2,  # Center-Right
            start_node_idx + 2 * nodes_per_slice,  # Bottom-Left
            start_node_idx + (2 * nodes_per_slice) + 1,  # Bottom-Middle
            start_node_idx + (2 * nodes_per_slice) + 2  # Bottom-Right
        ])
        return patch_nodes

    # ------------------------------------------
    # Right Face
    # ------------------------------------------

    def sample_random_right_face_2x2(self):
        patch_nodes = self.sample_random_left_face_2x2()
        offset = self.nelx * (self.nely + 1)
        return patch_nodes + offset

    def sample_random_right_face_3x3(self):
        patch_nodes = self.sample_random_left_face_3x3()
        offset = self.nelx * (self.nely + 1)
        return patch_nodes + offset

    # ------------------------------------------
    # Back Face
    # ------------------------------------------

    def sample_random_back_face_2x2(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        start_x = np.random.randint(0, self.nelx)
        start_z = np.random.randint(0, self.nelz)
        start_node_idx = (start_z * nodes_per_slice) + (start_x * ny)  # y is 0
        patch_nodes = np.array([
            start_node_idx,  # (x, z)
            start_node_idx + ny,  # (x+1, z) -> Jump by ny
            start_node_idx + nodes_per_slice,  # (x, z+1)
            start_node_idx + nodes_per_slice + ny  # (x+1, z+1)
        ])
        return patch_nodes

    def sample_random_back_face_3x3(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        start_x = np.random.randint(0, self.nelx - 1)
        start_z = np.random.randint(0, self.nelz - 1)
        start_node_idx = (start_z * nodes_per_slice) + (start_x * ny)  # y is 0
        patch_nodes = np.array([
            start_node_idx,  # (x, z)
            start_node_idx + ny,  # (x+1, z) -> Jump by ny
            start_node_idx + 2 * ny,  # (x+2, z)
            start_node_idx + nodes_per_slice,  # (x, z+1)
            start_node_idx + nodes_per_slice + ny,  # (x+1, z+1)
            start_node_idx + nodes_per_slice + 2 * ny,  # (x+2, z+1)
            start_node_idx + 2 * nodes_per_slice,  # (x, z+2)
            start_node_idx + 2 * nodes_per_slice + ny,  # (x+1, z+2)
            start_node_idx + 2 * nodes_per_slice + 2 * ny  # (x+2, z+2)
        ])
        return patch_nodes

    # ------------------------------------------
    # Front Face
    # ------------------------------------------

    def sample_random_front_face_2x2(self):
        patch_nodes = self.sample_random_back_face_2x2()
        offset = self.nely
        return patch_nodes + offset

    def sample_random_front_face_3x3(self):
        patch_nodes = self.sample_random_back_face_3x3()
        offset = self.nely
        return patch_nodes + offset

    # ------------------------------------------
    # Top Face
    # ------------------------------------------

    def sample_random_top_face_2x2(self):
        ny = self.nely + 1
        start_x = np.random.randint(0, self.nelx)
        start_y = np.random.randint(0, self.nely)
        start_node_idx = (start_y) + (start_x * (self.nely + 1))  # z=0
        patch_nodes = np.array([
            start_node_idx,  # (x, y)
            start_node_idx + 1,  # (x, y+1)
            start_node_idx + ny,  # (x+1, y)
            start_node_idx + ny + 1  # (x+1, y+1)
        ])
        return patch_nodes

    def sample_random_top_face_3x3(self):
        ny = self.nely + 1
        start_x = np.random.randint(0, self.nelx - 1)
        start_y = np.random.randint(0, self.nely - 1)
        start_node_idx = (start_y) + (start_x * (self.nely + 1))  # z=0
        patch_nodes = np.array([
            start_node_idx,  # (x, y)
            start_node_idx + 1,  # (x, y+1)
            start_node_idx + 2,  # (x, y+2)
            start_node_idx + ny,  # (x+1, y)
            start_node_idx + ny + 1,  # (x+1, y+1)
            start_node_idx + ny + 2,  # (x+1, y+2)
            start_node_idx + 2 * ny,  # (x+2, y)
            start_node_idx + 2 * ny + 1,  # (x+2, y+1)
            start_node_idx + 2 * ny + 2  # (x+2, y+2)
        ])
        return patch_nodes

    # ------------------------------------------
    # Bottom Face
    # ------------------------------------------

    def sample_random_bottom_face_2x2(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        patch_nodes = self.sample_random_top_face_2x2()
        offset = nodes_per_slice * self.nelz
        return patch_nodes + offset

    def sample_random_bottom_face_3x3(self):
        nx, ny, nz = self.nelx + 1, self.nely + 1, self.nelz + 1
        nodes_per_slice = nx * ny
        patch_nodes = self.sample_random_top_face_3x3()
        offset = nodes_per_slice * self.nelz
        return patch_nodes + offset








if __name__ == '__main__':
    save_path = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/bcs'
    n_samples = 1
    nelx, nely, nelz = 32, 32, 32
    bc_gen = BcGenerator(nelx, nely, nelz)
    bc_list = bc_gen.generate(save_path, n_samples)

    bc = bc_list[0]
    bc['volfrac'] = 0.2
    print(bc)
    print(bc['volfrac'])

    from D3.v2.thermoelastic3d_datagen import ThermoelasticTopologyOptimization3D
    volfrac = bc['volfrac']
    penal = 3.0
    rmin = 1.5
    el_weight = 1.0
    fname = 'test_design.npz'
    plot = True
    static_vf_init = 0.5

    l_ele = 1.0  # was 50
    opt = ThermoelasticTopologyOptimization3D(
        nelx, nely, nelz,
        volfrac, penal, rmin,
        lx=l_ele, ly=l_ele, lz=l_ele,
        iter_solve=True,
        fname=fname,
        el_weight=el_weight,
        plot=plot,
        boundary_conditions=bc,
        static_vf_init=static_vf_init,
    )
    opt.optimize(max_iter=50)






