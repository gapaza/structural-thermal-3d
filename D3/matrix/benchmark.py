import numpy as np
from scipy.sparse import load_npz
import time

from D3.utils.linear_solver import solve_spd_with_amg, solve_spd_with_amg2
from D3.utils.linear_solver import solve_with_spsolve
from D3.utils.linear_solver import solve_with_pardiso
from D3.utils.linear_solver import solve_with_mumps




def run():

    path_A = '/3D/matrix/store/A.npz'
    path_b = '/3D/matrix/store/b.npy'

    # path_A = '/home/gapaza/scratch/repos/structural-thermal-3d/matrix/store/A.npz'
    # path_b = '/home/gapaza/scratch/repos/structural-thermal-3d/matrix/store/b.npy'

    arr_A = load_npz(path_A)  # csr_matrix
    arr_b = np.load(path_b)   # ndarray

    t0 = time.time()

    print('Solving system')

    ### MAC
    # x = solve_spd_with_amg(arr_A, arr_b)  # 36 seconds (mac)
    x = solve_spd_with_amg2(arr_A, arr_b)  #
    # x = solve_with_spsolve(arr_A, arr_b)    # many minutes (mac)

    ### HPC
    # x = solve_with_spsolve(arr_A, arr_b)    # many minutes
    # x = solve_with_pardiso(arr_A, arr_b)



    t_elapsed = time.time() - t0
    print(f"Solved in {t_elapsed:.4f} seconds.")














if __name__ == "__main__":
    run()



