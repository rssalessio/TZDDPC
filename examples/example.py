# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from szddpc import SZDDPC, Data, SystemZonotopes
from utils import generate_trajectories
from pydatadrivenreachability import Zonotope

# Define the loss function
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


# Plant
# In this example we consider the three-pulley 
# system analyzed in the original VRFT paper:
# 
# "Virtual reference feedback tuning: 
#      a direct method for the design offeedback controllers"
# -- Campi et al. 2003

dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = scipysig.TransferFunction(num, den, dt=dt).to_ss()
dim_x, dim_u = sys.B.shape


# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0. * np.diag([1] * dim_x))
U = Zonotope([1] * dim_u, 3 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.005 * np.ones((dim_x, 1)))
V = Zonotope([0] * dim_x, 0.002 * np.ones((dim_x, 1)))
Y = Zonotope([1] * dim_x, np.diag(2*np.ones(dim_x)))
AV = V * sys.A
zonotopes = SystemZonotopes(X0, U, Y, W, V, AV)

num_trajectories = 5
num_steps_per_trajectory = 200
horizon =4

data = generate_trajectories(sys, X0, U, W, V, num_trajectories, num_steps_per_trajectory)

# Build DPC
szddpc = SZDDPC(data)

x = X0.sample().flatten()


trajectory = [x]
problem = szddpc.build_problem(zonotopes, horizon, loss_callback, constraints_callback)
for n in range(100):
    print(f'Step {n}')
    #import pdb
    #pdb.set_trace()
    
    result, info = szddpc.solve(x, verbose=False,warm_start=True)
    u = info['u_optimal']
    #print(u)
    #import pdb
    #pdb.set_trace()
    z = sys.A @ x +  np.squeeze(sys.B *u[0]) + W.sample()

    # We assume C = I
    x = (z + V.sample()).flatten()
    trajectory.append(x)

trajectory = np.array(trajectory)
for i in range(dim_x):
    plt.plot(trajectory[:,i], label=f'x{i}')

plt.grid()
plt.legend()
plt.show()
# print(info)
# plt.figure()
# plt.margins(x=0, y=0)

# # Simulate for different values of T
# for T in T_list:
#     sys.reset()
#     # Generate initial data and initialize DeePC
#     data = sys.apply_input(u = np.random.normal(size=T).reshape((T, 1)), noise_std=0)
#     deepc = DeePC(data, Tini = T_INI, horizon = HORIZON)

#     # Create initial data
#     data_ini = Data(u = np.zeros((T_INI, 1)), y = np.zeros((T_INI, 1)))
#     sys.reset(data_ini = data_ini)

#     deepc.build_problem(
#         build_loss = loss_callback,
#         build_constraints = constraints_callback,
#         lambda_g = LAMBDA_G_REGULARIZER,
#         lambda_y = LAMBDA_Y_REGULARIZER)

#     for idx in range(300):
#         # Solve DeePC
#         u_optimal, info = deepc.solve(data_ini = data_ini, warm_start=True)


#         # Apply optimal control input
#         _ = sys.apply_input(u = u_optimal[:s, :], noise_std=0)

#         # Fetch last T_INI samples
#         data_ini = sys.get_last_n_samples(T_INI)

#     # Plot curve
#     data = sys.get_all_samples()
#     plt.plot(data.y[T:], label=f'$s={s}, T={T}, T_i={T_INI}, N={HORIZON}$')

# plt.ylim([0, 2])
# plt.xlabel('Step')
# plt.ylabel('y')
# plt.title('Closed loop output')
# plt.legend(fancybox=True, shadow=True)
# plt.grid()
# plt.show()