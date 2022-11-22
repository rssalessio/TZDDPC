# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
import time
from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from tzddpc import TZDDPC, Data, SystemZonotopes
from utils import generate_trajectories
from pyzonotope import Zonotope
from pyzpc import ZPC, Data as DataZPC, SystemZonotopes as ZonotopesZPC
from scipy.linalg import solve_discrete_are

# Define the loss function
def loss_callback(u: cp.Variable, y: cp.Variable) -> Expression:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    cost = 0
    for i in range(horizon):
        cost += cp.norm(y[i,0]-1,p=2)
    return  cost

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

zpc_times = []
tzddpc_times = []
zpc_data = []
tzddpc_data = []

# Define zonotopes and generate data
X0 = Zonotope([0] * dim_x, 0. * np.ones((dim_x,1)))
U = Zonotope([1] * dim_u, 3 * np.ones((dim_u,1)))
W = Zonotope([0] * dim_x, 1e-1* np.ones((dim_x, 1)))
X = Zonotope([1] * dim_x, 2*np.ones((dim_x, 1)))
zonotopes = SystemZonotopes(X0, U, X, W)
W_vertices = W.compute_vertices()
num_W_vertices = len(W_vertices)
num_trajectories = 1
num_steps_per_trajectory = 400


for id_run in range(5):
    data = generate_trajectories(sys, X0, U, W,  num_trajectories, num_steps_per_trajectory)

    # Build TZDDPC
    tzddpc = TZDDPC(data)
    tzddpc.build_zonotopes_theta(zonotopes)
    x0 = X0.sample().flatten()

    total_steps  = 200
    x = [x0]
    xbar = [x[-1].copy()]
    e = [np.zeros_like(x[-1])]
    Ze = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]
    N = 2


    
    start_time_tzddpc = time.time()
    tzddpc.build_problem(N, loss_callback, constraints_callback)
    for t in range(total_steps):
        result, v, xbark, Zek = tzddpc.solve(
            xbar[-1],
            e[-1],
            verbose=False
        )
        # print(f'[{t}] x: {x[-1]} - xbar: {xbar[-1]} - v: {v[0]}')
        

        xbar.append(xbark[1])
        u = tzddpc.theta.K @ e[-1] + v[0]
        x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W.sample()#[np.random.choice(num_W_vertices)]
        x.append(x_next.flatten())
        e.append(x[-1] - xbar[-1])
        
        Zek = Zek.Z.value
    end_time_tzddpc = time.time() - start_time_tzddpc
    

    x_tzddpc = np.array(x)
    tzddpc_data.append(x_tzddpc)
    tzddpc_times.append(end_time_tzddpc)
    np.save(f'results/pulley.xtzddpc.{id_run}', x_tzddpc)
    x = [x0]

    # Acenter, Bcenter = tzddpc.Mdata.center[:, :dim_x], tzddpc.Mdata.center[:, dim_x:]
    # P = solve_discrete_are(Acenter, Bcenter, np.eye(dim_x), np.eye(dim_u))
    # Klqr = -np.linalg.inv(np.eye(dim_u) + Bcenter.T @ P @ Bcenter) @ (Bcenter.T @ P @ Acenter)
    # for t in range(total_steps):
    #     u = Klqr @ x[-1]# + 1
    #     x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W.sample()
    #     x.append(x_next.flatten())

    # x_lqr = np.array(x)




    # BUILD ZPC
    print('BUILDING ZPC')
    zpc_zonotopes = ZonotopesZPC(X0, U, X, W, Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))), Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))))
    zpc = ZPC(DataZPC(data.u, data.x))


    start_time_zpc = time.time()
    problem = zpc.build_problem(zpc_zonotopes, N, loss_callback, constraints_callback)
    x = [x0]

    for n in range(total_steps):
        #print(f'Solving step {n}')
        result, info = zpc.solve(x[-1], verbose=False,warm_start=True)
        u = info['u_optimal']
        x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u[0]) + W.sample()# W_vertices[np.random.choice(num_W_vertices)]
        x.append(x_next.flatten())
    end_time_zpc = time.time() - start_time_zpc
    x_zpc = np.array(x)


    zpc_times.append(end_time_zpc)
    zpc_data.append(x_zpc)

    np.save(f'results/pulley.xzpc.{id_run}', x_zpc)
    print(f'ZPC: {round(end_time_zpc/60,2)}')
    print(f'TZDDPC: {round(end_time_tzddpc/60,2)}')

np.save('results/pulley.xzpc', zpc_data)
np.save('results/pulley.tzddpc', tzddpc_data)
np.save('results/pulley.zpc_times', zpc_times)
np.save('results/pulley.tzddpc_times', tzddpc_times)
plt.plot(x_tzddpc[:,0], label=f'TZDDPC - $x_1$')
plt.plot(x_zpc[:,0], label=f'ZPC - $x_1$')
plt.grid()
plt.legend()
plt.show()

