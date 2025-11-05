from scipy.sparse.linalg import LinearOperator, cg
from scipy.sparse.linalg import spsolve
import pyamg


# AMG is still the fastest solver for large sparse linear systems
def solve_spd_with_amg(A, b, tol=1e-8, maxiter=None):
    ml = pyamg.smoothed_aggregation_solver(A)  # or ruge_stuben_solver for pure Poisson
    M = ml.aspreconditioner()                   # right preconditioner
    x, info = cg(A, b, rtol=tol, M=M, maxiter=maxiter)
    if info != 0:
        # fallback (rare): one V-cycle polish + CG again or direct
        x = ml.solve(b, x0=x, tol=tol)          # a few V-cycles
    return x

def solve_spd_with_amg2(A, b, tol=1e-8, maxiter=200):
    # ml = pyamg.rootnode_solver(A, smooth='energy')
    ml = pyamg.smoothed_aggregation_solver(A, smooth='energy')
    residuals = []
    x = ml.solve(b, tol=1e-10, maxiter=None, residuals=residuals)
    return x



def solve_with_spsolve(A, b):
    x = spsolve(A, b)
    return x



# TODO: pardiso is not implemented on arm64 yet
def solve_with_pardiso(A, b):
    from pypardiso import spsolve as pardiso_spsolve
    x = pardiso_spsolve(A, b)
    return x



def solve_with_mumps(A, b):
    pass







