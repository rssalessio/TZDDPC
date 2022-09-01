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
from pydatadrivenreachability import Zonotope
from ignored.constants_system import A,B,C,D, X0, U, W, X, sys, dim_x, dim_u, loss_callback, constraints_callback


# DeePC paramters
T_INI = 5                   # Size of the initial set of data
T_DATA = 1000               # Number of data points used to estimate the system
HORIZON = 50                # Horizon length
LAMBDA_G_REGULARIZER = 10 # g regularizer (see DeePC paper, eq. 8)
LAMBDA_Y_REGULARIZER = 1    # y regularizer (see DeePC paper, eq. 8)


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