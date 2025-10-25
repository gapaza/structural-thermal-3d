import numpy as np
from scipy.sparse import load_npz, csr_matrix
import time

from utils.linear_solver import solve_spd_with_amg
from utils.linear_solver import solve_with_spsolve




def run():

    path_A = '/Users/gapaza/repos/ideal/structural-thermal-3d/matrix/store/A.npz'
    path_b = '/Users/gapaza/repos/ideal/structural-thermal-3d/matrix/store/b.npy'

    arr_A = load_npz(path_A)  # csr_matrix
    arr_b = np.load(path_b)   # ndarray

    t0 = time.time()

    print('Solving system')
    # x = solve_spd_with_amg(arr_A, arr_b)  # 36 seconds
    x = solve_with_spsolve(arr_A, arr_b)    # many minutes

    t_elapsed = time.time() - t0
    print(f"Solved in {t_elapsed:.4f} seconds.")
























if __name__ == "__main__":
    run()



