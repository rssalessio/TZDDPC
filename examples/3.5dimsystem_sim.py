# To run this example you also need to install matplotlib
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from tzddpc import TZDDPC, Data, SystemZonotopes
from utils import generate_trajectories
from pyzonotope import Zonotope

def loss_callback(u: cp.Variable, x: cp.Variable) -> cp.Expression:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]

    cost = 0
    for i in range(horizon):
        cost += 1e9* cp.norm(x[i,1]-2,p=2) +  1e-1*cp.norm(u[i], p=2)
    return  cost


def constraints_callback(u: cp.Variable, x: cp.Variable) -> List[Constraint]:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]
    # Define a list of additional input/output constraints
    return [x[:, 1] <= 10, x[:,1] >=2] #, x >= -2, x <= 4, u >= -6, u <= 6.]


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
X0 = Zonotope([-2, 4, 3, -2.5, 5.5], 0 * np.diag([1] * dim_x))
U = Zonotope([7] * dim_u,  100 * np.diag([1] * dim_u))
W = Zonotope([0] * dim_x, 0.1* np.ones((dim_x, 1)))

Id = 20 * np.ones((dim_x, 1))
Id[1] = 19
X = Zonotope([1, 20, 1, 1, 1], Id)
W_vertices = W.compute_vertices()
num_W_vertices = len(W_vertices)
zonotopes = SystemZonotopes(X0, U, X, W)

num_trajectories = 1
num_steps_per_trajectory = 400
horizon =10

data = generate_trajectories(sys, X0, U, W,  num_trajectories, num_steps_per_trajectory)

# Build DPC
tzddpc = TZDDPC(data)
tzddpc.build_zonotopes_theta(zonotopes)
x0 = X0.sample().flatten()


x = [x0]
xbar = [x[-1].copy()]
e = [np.zeros_like(x[-1])]
Ze = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]

tzddpc.build_problem(2, loss_callback, constraints_callback)

for t in range(200):
    result, v, xbark, Zek = tzddpc.solve(
        xbar[-1],
        e[-1],
        verbose=False
    )
    print(f'[{t}] x: {x[-1]} - xbar: {xbar[-1]} - v: {v[0]}')
    

    xbar.append(xbark[1])
    u = tzddpc.theta.K @ e[-1] + v[0]
    x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W.sample()# W_vertices[np.random.choice(num_W_vertices)]
    x.append(x_next.flatten())
    e.append(x[-1] - xbar[-1])
    
    Zek = Zek.Z.value
    Ze.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar[-1])

x = np.array(x)
plt.plot(x[:,1], label=f'$x_1$')

plt.grid()
plt.legend()
plt.show()
import pdb
pdb.set_trace()