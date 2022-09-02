# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydeepc import DeePC
from pydeepc.utils import Data
from utils import generate_trajectories
from pyzonotope import Zonotope

def loss_callback(u: cp.Variable, x: cp.Variable) -> cp.Expression:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]

    cost = 0
    for i in range(horizon):
        cost += cp.norm(x[i,1] - 1) +  1e-2*cp.norm(u[i], p=1)
    return  cost


def constraints_callback(u: cp.Variable, x: cp.Variable) -> List[Constraint]:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]
    # Define a list of additional input/output constraints
    return [x[:, 1] <= 10, x[:,1]>=1]#x >= -2, x <= 4, u >= -6, u <= 6.]

A = np.array(
    [[-1, -4, 0, 0, 0],
     [4, -1, 0, 0, 0],
     [0, 0, -3, 1, 0],
     [0, 0, -1, -3, 0],
     [0, 0, 0, 0, -2]])
B = np.ones((5, 1))
dim_x, dim_u = B.shape
dt = 0.05
A,B,C,D,_ = scipysig.cont2discrete(system=(A,B,np.eye(dim_x),0*B), dt = dt)
sys = scipysig.StateSpace(A,B,C,D)

# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0 * np.diag([1] * dim_x))
U = Zonotope([7] * dim_u,  19 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.01* np.ones((dim_x, 1)))
X = Zonotope([1] * dim_x, 100 * np.ones((dim_x, 1)))

# DeePC paramters
T_INI = 1                   # Size of the initial set of data
T_DATA = 400               # Number of data points used to estimate the system
HORIZON = 50                # Horizon length
LAMBDA_G_REGULARIZER = 1 # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 0    # y regularizer (see DeePC paper, eq. 8)


data = generate_trajectories(sys, X0, U, W, 1, T_DATA)

plt.figure()
plt.margins(x=0, y=0)

# Generate initial data and initialize DeePC

deepc = DeePC(Data(data.u, data.x), Tini = T_INI, horizon = HORIZON)

# Create initial data

deepc.build_problem(
    build_loss = loss_callback,
    build_constraints = constraints_callback,
    lambda_g = LAMBDA_G_REGULARIZER,
    lambda_y = LAMBDA_Y_REGULARIZER)

TRAJECTORY_HORIZON = 50
trajectory_u = np.zeros((TRAJECTORY_HORIZON+T_INI, dim_u))
trajectory_x = np.zeros((TRAJECTORY_HORIZON+1+T_INI, dim_x))
data_ini = Data(u = np.zeros((T_INI, dim_u)), y = np.zeros((T_INI, dim_x)))

for idx in range(T_INI, TRAJECTORY_HORIZON+T_INI):
    print(f'Solving step {idx}')
    u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True)

    trajectory_u[idx,:] = u_optimal[0,:]

    trajectory_x[idx+1,:] = (sys.A @ trajectory_x[idx,:] +  np.squeeze(sys.B @ trajectory_u[idx]) + W.sample()).flatten()

    # Fetch last T_INI samples
    data_ini = Data(u = trajectory_u[idx - T_INI: idx,:], y = trajectory_x[idx + 1 - T_INI:idx+1,:])


plt.plot(trajectory_x[:,1])
plt.grid()
plt.show()

# for n in range(70):
#     print(f'Step {n} - x: {x_traj[-1]} -  e: {e_traj[-1]}')
#     res, v, xbar, Ze = szddpc.solve(xbar_traj[-1], e_traj[-1], 10, zonotopes.sigma, loss_callback, constraints_callback, solver=cp.ECOS, warm_start=False, verbose=False)
#     Ze = Zonotope(Ze.center.value, Ze.generators.value)

#     u = theta.K @ (e_traj[-1] + xbar_traj[-1]) + v[0]

#     xbar_traj.append(xbar[1])
#     x = (sys.A @ x +  np.squeeze(sys.B @u) + W.sample()).flatten()
    
#     e_traj.append(x - xbar_traj[-1])
#     x_traj.append(x)
#     res, err_proj, beta = Ze.projection(e_traj[-1])
#     if res > 0:
#         print(f'Error: {res} - {np.abs(e_traj[-1]-err_proj)}')

# # Plot curve
# data = sys.get_all_samples()
# plt.plot(data.y[T:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')

# plt.ylim([0, 2])
# plt.xlabel('Step')
# plt.ylabel('y')
# plt.title('Closed loop output')
# plt.legend(fancybox=True, shadow=True)
# plt.grid()
# plt.show()