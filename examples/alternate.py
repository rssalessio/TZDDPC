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

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    ref = np.array([[1,0,0,0]]*horizon)
    # import pdb
    # pdb.set_trace()
    # Sum_t ||y_t - r_t||^2
    cost = 0
    for i in range(horizon):
        cost += 1*cp.norm(y[i,1] - 1)
    return  cost #100*cp.sum(cp.norm(y[1:] - ref, p=2, axis=1))

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return []

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


# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 1 * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u,  5 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 1* np.ones((dim_x, 1)))
X = Zonotope([1] * dim_x, 2*np.diag(np.ones(dim_x)))

dt = 0.05
# num = [0.28261, 0.50666]
# den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
# sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()

sys = scipysig.StateSpace(A,B,C,D)
# dim_x, dim_u = sys.B.shape



# # Define zonotopes and generate data
# X0 = Zonotope([0] * dim_x, 0. * np.diag([1] * dim_x))
# U = Zonotope([0] * dim_u, 5 * np.diag([1] * dim_u))
# W = Zonotope([0] * dim_x, 0.3 * np.diag(np.ones((dim_x))))
# X = Zonotope([1] * dim_x, 2*np.diag(np.ones(dim_x)))
sigma = Zonotope([0] * dim_x, 0*np.diag([1] * dim_x))
zonotopes = SystemZonotopes(X0, U, X, W, sigma)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon =40

data = generate_trajectories(sys, X0, U, W, num_trajectories, num_steps_per_trajectory)

# Build DPC
szddpc = SZDDPC(data)
theta, M = szddpc.build_zonotopes_theta(zonotopes, tol=1e-2, num_initial_points=1)
import pdb
pdb.set_trace()
A0,B0 = A,B
A,B = M.center[:, :dim_x], M.center[:, dim_x:]
Acl = A + B @ theta.K
DeltaA0 = A0 - A
DeltaB0 = B0 - B

DeltaM =   (-1*M) +M.center
XU = X.cartesian_product(U)
DeltaMXU = DeltaM * XU

Z_e = [Zonotope([0] * dim_x, np.zeros((dim_x, 1)))]
for i in range(200):
    Z_e[i] = Z_e[i].reduce(50)
    
    Z_e.append(Z_e[i] * (Acl) +DeltaMXU + W)
    print(Z_e[-1].interval)

import pdb
pdb.set_trace()






x = X0.sample().flatten()

x_traj = [x]
xbar_traj = [x]
e_traj = [np.zeros_like(x)]

for n in range(70):
    print(f'Step {n} - x: {x_traj[-1]} -  e: {e_traj[-1]}')
    res, v, xbar, Ze = szddpc.solve(xbar_traj[-1], e_traj[-1], 10, zonotopes.sigma, loss_callback, constraints_callback, solver=cp.ECOS, warm_start=False, verbose=False)
    Ze = Zonotope(Ze.center.value, Ze.generators.value)

    u = theta.K @ (e_traj[-1] + xbar_traj[-1]) + v[0]

    xbar_traj.append(xbar[1])
    x = (sys.A @ x +  np.squeeze(sys.B @u) + W.sample()).flatten()
    
    e_traj.append(x - xbar_traj[-1])
    x_traj.append(x)
    res, err_proj, beta = Ze.projection(e_traj[-1])
    if res > 0:
        print(f'Error: {res} - {np.abs(e_traj[-1]-err_proj)}')



plt.plot(np.array(x_traj)[:, 1])
plt.grid()
plt.show()