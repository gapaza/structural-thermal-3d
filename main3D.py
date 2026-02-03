import config
import os
import argparse
import time
import multiprocessing
from copy import deepcopy
from tqdm import tqdm

from D3.v2.thermoelastic3d_datagen import ThermoelasticTopologyOptimization3D

NELX = 32
NELY = 32
NELZ = 32

def optimize(bc):
    OPT_STEPS = 200

    # Structural Optimization
    bc_struct = deepcopy(bc)
    bc_struct['records_path'] = bc_struct['struct_path']
    volfrac = bc_struct['volfrac']
    penal = 3.0
    rmin = 1.5
    el_weight = 1.0
    fname = 'test_design_struct.npz'
    plot = False
    static_vf_init = 0.5
    l_ele = 1.0
    opt = ThermoelasticTopologyOptimization3D(
        NELX, NELY, NELZ,
        volfrac, penal, rmin,
        lx=l_ele, ly=l_ele, lz=l_ele,
        iter_solve=True,
        fname=fname,
        el_weight=el_weight,
        plot=plot,
        boundary_conditions=bc_struct,
        static_vf_init=static_vf_init,
    )
    opt.optimize(max_iter=OPT_STEPS)

    # Thermal Optimization
    bc_therm = deepcopy(bc)
    bc_therm['records_path'] = bc_therm['therm_path']
    volfrac = bc_therm['volfrac']
    penal = 3.0
    rmin = 1.5
    el_weight = 0.0
    fname = 'test_design_therm.npz'
    plot = False
    static_vf_init = 0.5
    l_ele = 1.0
    opt = ThermoelasticTopologyOptimization3D(
        NELX, NELY, NELZ,
        volfrac, penal, rmin,
        lx=l_ele, ly=l_ele, lz=l_ele,
        iter_solve=True,
        fname=fname,
        el_weight=el_weight,
        plot=plot,
        boundary_conditions=bc_therm,
        static_vf_init=static_vf_init,
    )
    opt.optimize(max_iter=OPT_STEPS)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a dataset using multiple processes with configurable parameters.'
    )
    parser.add_argument(
        '--num-procs',
        type=int,
        default=5,
        help='Number of processes to use (default: 5)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of samples to generate (default: 5)'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/bcs',
        help='Directory to save the dataset (default: /home/gapaza/scratch/datasets/thermoelastic2dv000)'
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    num_procs = args.num_procs
    num_samples = args.samples
    save_dir = args.save_dir


    # 1. Generate boundary conditions
    from D3.v2.bc_generator import BcGenerator
    bc_gen = BcGenerator(NELX, NELY, NELZ)
    bc_list = bc_gen.generate(save_dir, num_samples)

    # 2. Run with multiprocessing
    with multiprocessing.Pool(processes=num_procs) as pool:
        # imap_unordered yields results as soon as they're ready.
        results = list(tqdm(pool.imap_unordered(optimize, bc_list), total=len(bc_list)))












