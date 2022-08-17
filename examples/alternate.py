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

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0. * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u, 3 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 1 * np.ones((dim_x, 1)))
V = Zonotope([0] * dim_x, 0.02 * np.ones((dim_x, 1)))
Y = Zonotope([1] * dim_x, np.diag(0.1*np.ones(dim_x)))
AV = V * sys.A
zonotopes = SystemZonotopes(X0, U, Y, W, V, AV)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon =10

data = generate_trajectories(sys, X0, U, W, V, num_trajectories, num_steps_per_trajectory)

# Build DPC
szddpc = SZDDPC(data)

Msigma = szddpc.build_zonotopes(zonotopes)
compute_theta(Msigma, Msigma.center[:, :dim_x], Msigma.center[:, dim_x:])
compute_theta2(Msigma, Msigma.center[:, :dim_x], Msigma.center[:, dim_x:])