# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from szddpc import SZDDPC, Data, SystemZonotopes, compute_theta,compute_theta2
from utils import generate_trajectories
from pydatadrivenreachability import Zonotope
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.array([[1,0,0,0]]*horizon)
    # import pdb
    # pdb.set_trace()
    # Sum_t ||y_t - r_t||^2
    cost = 0
    for i in range(horizon):
        cost += 100*cp.norm(y[i,0] - 1)
    return  cost #100*cp.sum(cp.norm(y[1:] - ref, p=2, axis=1))

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return []

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0. * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u, 3 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.1 * np.ones((dim_x, 1)))
X = Zonotope([1] * dim_x, np.diag(0.1*np.ones(dim_x)))
sigma = Zonotope([0] * dim_x, np.diag([1] * dim_x))
zonotopes = SystemZonotopes(X0, U, X, W, sigma)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon =10

data = generate_trajectories(sys, X0, U, W, num_trajectories, num_steps_per_trajectory)

# Build DPC
szddpc = SZDDPC(data)

# Msigma = szddpc.build_zonotopes(zonotopes)
# szddpc.compute_theta()
# import pdb
# pdb.set_trace()
szddpc.build_problem(zonotopes, 10, loss_callback, constraints_callback, tol=1e-2, num_initial_points=1)