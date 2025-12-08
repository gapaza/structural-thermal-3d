"""This module contains the MMA subroutine used in the thermoelastic2d problem."""

"""
number of constraints: 1
number of design variables: 400
x val shape: (400, 1)
f0val shape: ()
df0dx shape: (400, 1)
fval shape: ()
dfdx shape: (1, 400)
low shape: (400, 1)
upp shape: (400, 1)
a shape: (1, 1)
c shape: (1, 1)
d shape: (1, 1)
xold1 shape: (400, 1)
xold2 shape: (400, 1)
xmin shape: (400, 1)
xmax shape: (400, 1)

number of constraints: 1
number of design variables: 4096
x val shape: (4096, 1)
f0val shape: ()
df0dx shape: (4096, 1)
fval shape: ()
dfdx shape: (1, 4096)
low shape: (4096, 1)
upp shape: (4096, 1)
a shape: (1, 1)
c shape: (1, 1)
d shape: (1, 1)
xold1 shape: (4096, 1)
xold2 shape: (4096, 1)
xmin shape: (4096, 1)
xmax shape: (4096, 1)
"""


from dataclasses import dataclass

from mmapy import mmasub as external_mmasub
import numpy as np
from numpy.typing import NDArray

RESIDUAL_MAX_VAL = 0.9
ITERATION_MAX = 500
ITERATION_MAX_SMALL = 50
ITERATION_ASYM_MAX = 2.5

@dataclass(frozen=True)
class MMAInputs:
    """Dataclass encapsulating all input parameters for the MMA subroutine."""

    m: int
    """The number of constraints"""
    n: int
    """The number of design variables"""
    iterr: int
    """The current iteration number"""
    xval: NDArray[np.float64]
    """The flattened array of design variables: shape (n,)"""
    xmin: float
    """The lower bounds on the design variables"""
    xmax: float
    """The upper bounds on the design variables"""
    xold1: NDArray[np.float64]
    """The previous design variables at iteration k-1: shape (n, 1)"""
    xold2: NDArray[np.float64]
    """The design variables at iteration k-2: shape (n, 1)"""
    df0dx: NDArray[np.float64]
    """The gradients of the objective function at xval: shape (n,)"""
    fval: NDArray[np.float64]
    """The value of the constraint functions evaluated at xval: shape (m,)"""
    dfdx: NDArray[np.float64]
    """The gradients of the constraint functions at xval: shape (m, n)"""
    low: NDArray[np.float64]
    """The lower asymptotes from the previous iteration: shape (n,)"""
    upp: NDArray[np.float64]
    """The upper asymptotes from the previous iteration: shape (n,)"""
    a0: float
    """The constants a_0 in the term a_0*z"""
    a: NDArray[np.float64]
    """the constants a_i in the terms a_i*z"""
    c: NDArray[np.float64]
    """the constants c_i in the terms c_i*y_i"""
    d: NDArray[np.float64]
    """the constants d_i in the terms 0.5*d_i*(y_i)^2"""
    f0val: float = 0.0
    """The value of the objective function at xval"""


def mmasub(inputs: MMAInputs) -> NDArray[np.float64]:
    """Perform one MMA iteration to solve a nonlinear programming problem using the GCMMA-MMA-Python library.

    Minimize:
        f_0(x) + a_0 * z + sum(c_i * y_i + 0.5 * d_i * (y_i)^2)

    Subject to:
        f_i(x) - a_i * z - y_i <= 0,    i = 1,...,m
        xmin_j <= x_j <= xmax_j,        j = 1,...,n
        z >= 0, y_i >= 0,               i = 1,...,m

    Parameters:
        inputs (MMAInputs): A dataclass encapsulating all input parameters.

    Returns:
        xmma (NDArray[np.float64]): the updated design variables.
        low (NDArray[np.float64]): the updated lower bounds.
        upp (NDArray[np.float64]): the updated upper bounds.
    """
    # Unpack parameters from the dataclass.
    m = int(inputs.m)
    n = int(inputs.n)
    iterr = int(inputs.iterr)
    # xval = np.expand_dims(inputs.xval, axis=-1)
    xval = np.reshape(inputs.xval, (-1, 1))
    # xval = inputs.xval
    xmin = np.full((n, 1), inputs.xmin)
    xmax = np.full((n, 1), inputs.xmax)
    # xold1 = np.expand_dims(inputs.xold1, axis=-1)
    # xold2 = np.expand_dims(inputs.xold2, axis=-1)
    xold1 = np.reshape(inputs.xold1, (-1, 1))
    xold2 = np.reshape(inputs.xold2, (-1, 1))
    f0val = inputs.f0val
    df0dx = np.expand_dims(inputs.df0dx, axis=1)
    # fval = inputs.fval
    fval = np.squeeze(inputs.fval)
    dfdx = inputs.dfdx
    # low = np.expand_dims(inputs.low, axis=1)
    # upp = np.expand_dims(inputs.upp, axis=1)
    low = np.reshape(inputs.low, (-1, 1))
    upp = np.reshape(inputs.upp, (-1, 1))
    a0 = inputs.a0
    a = np.expand_dims(inputs.a, axis=1)
    c = np.expand_dims(inputs.c, axis=1)
    d = np.expand_dims(inputs.d, axis=1)

    # print('number of constraints:', m)
    # print('number of design variables:', n)
    # print('x val shape:', xval.shape)
    # print('f0val shape:', f0val.shape)
    # print('df0dx shape:', df0dx.shape)
    # print('fval shape:', fval.shape)
    # print('dfdx shape:', dfdx.shape)
    # print('low shape:', low.shape)
    # print('upp shape:', upp.shape)
    # print('a shape:', a.shape)
    # print('c shape:', c.shape)
    # print('d shape:', d.shape)
    # print('xold1 shape:', xold1.shape)
    # print('xold2 shape:', xold2.shape)
    # print('xmin shape:', xmin.shape)
    # print('xmax shape:', xmax.shape)
    # exit(0)



    # MMA parameters
    raa0 = 1e-5
    move = 0.2
    albefa = 0.1
    asyinit = 0.01
    asyincr = 1.2
    asydecr = 0.7
    asymax = 0.2
    asymin = 0.01

    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, up = external_mmasub(
        m,
        n,
        iterr,
        xval,
        xmin,
        xmax,
        xold1,
        xold2,
        f0val,
        df0dx,
        fval,
        dfdx,
        low,
        upp,
        a0,
        a,
        c,
        d,
        move=move,
        asyinit=asyinit,
        asydecr=asydecr,
        asyincr=asyincr,
        asymin=asymin,
        asymax=asymax,
        raa0=raa0,
        albefa=albefa,
    )

    # xmma = np.reshape(xmma, (-1, 1))
#     """
#     xmma return shape: (4096, 1)
# low return shape: (4096, 1)
# upp return shape: (4096, 1)"""

    # print('xmma return shape:', xmma.shape)
    # print('low return shape:', low.shape)
    # print('upp return shape:', upp.shape)

    return xmma, low, up