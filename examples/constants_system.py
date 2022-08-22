import numpy as np
import scipy.signal as scipysig
from pydatadrivenreachability import Zonotope
import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from typing import List

A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.ones((5, 1))
C = np.array([1, 0, 0, 0, 0])
D = np.array([0])

dim_x = A.shape[0]
dim_u = 1
dt = 0.05
A,B,C,D,_ = scipysig.cont2discrete(system=(A,B,C,D), dt = dt)
sys = scipysig.StateSpace(A,B,C,D)

# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 1 * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u,  5 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 1e-1* np.ones((dim_x, 1)))
X = Zonotope([1] * dim_x, 2*np.diag(np.ones(dim_x)))


def loss_callback(u: cp.Variable, y: cp.Variable) -> cp.Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]

    cost = 0
    for i in range(horizon):
        cost += 1000*cp.norm(y[i,1] - 1)
    return  cost

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return [y >= -2, y <= 4, u >= -6, u <= 6.]