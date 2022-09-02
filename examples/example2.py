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
    return [x[:, 1] <= 10, x[:,1] >=0]#x >= -2, x <= 4, u >= -6, u <= 6.]


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

zonotopes = SystemZonotopes(X0, U, X, W)

num_trajectories = 1
num_steps_per_trajectory = 400
horizon =10

data = generate_trajectories(sys, X0, U, W,  num_trajectories, num_steps_per_trajectory)

# Build DPC
szddpc = SZDDPC(data)
szddpc.build_zonotopes_theta(zonotopes)
x0 = X0.sample().flatten()


x = [x0]
xbar = [x[-1].copy()]
e = [np.zeros_like(x[-1])]
Ze = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]

szddpc.build_problem(2, loss_callback, constraints_callback)

for t in range(100):
    result, v, xbark, Zek = szddpc.solve(
        xbar[-1],
        e[-1],
        verbose=False
    )
    

    xbar.append(xbark[1])
    u = szddpc.theta.K @ x[-1] + v[0]
    print(f'[{t}] x: {x[-1]} - xbar: {xbar[-1]} - v: {v[0]} - u: {u} - ubar: {szddpc.theta.K @ xbar[-1] + v[0]} - Ke: {szddpc.theta.K @ e[-1]} - Kx {szddpc.theta.K @ x[-1] }')
    x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W.sample()
    x.append(x_next.flatten())
    e.append(x[-1] - xbar[-1])
    
    Zek = Zek.Z.value
    Ze.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar[-1])

x = np.array(x)
for i in range(dim_x):
    plt.plot(x[:,i], label=f'x{i}')

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