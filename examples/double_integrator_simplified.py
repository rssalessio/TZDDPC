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
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
np.random.seed(25)


# Define the loss function
def loss_callback(u: cp.Variable, x: cp.Variable) -> Expression:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]

    cost = 0
    for i in range(horizon):
        cost += cp.norm(x[i,:], p=2) ** 2 + 1e-2 * cp.norm(u[i], p=1)
    return  cost

# Define additional constraints
def constraints_callback(u: cp.Variable, y: cp.Variable) -> List[Constraint]:
    horizon, M, P = u.shape[0], u.shape[1], y.shape[1]
    # Define a list of additional input/output constraints
    return []

# Double integrator
dim_x = 2
dim_u = 1
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5],[1]])
C = np.eye(dim_x)
D = np.zeros((dim_x, dim_u))
sys = scipysig.StateSpace(A,B,C,D)
dim_x, dim_u = sys.B.shape

num_trajectories = 1
num_steps_per_trajectory = 200
total_steps = 9
horizon = min(10, total_steps)

eps = [0.1] * horizon

# Define zonotopes and generate data
X0 = Zonotope([-5, -2], 1e-5 * np.eye(dim_x))
U = Zonotope([0], 1*np.ones((1,1)))
W = Zonotope(np.zeros(dim_x), 0.05* np.array([[1, 0.5], [0.5, 1]]))
X = Zonotope([-2, -1], 2*np.diag([4,3]))
Zsigma = [Zonotope(np.zeros(dim_x), eps_k* np.eye(dim_x)) for eps_k in eps]

zonotopes = SystemZonotopes(X0, U, X, W)


data = generate_trajectories(sys, X0, U, W, num_trajectories, num_steps_per_trajectory)

x0 = X0.sample().flatten()

# Build DPC
szddpc = SZDDPC(data)

szddpc.build_zonotopes_theta(zonotopes)


x_full = [x0]
xbar_full = [x_full[-1].copy()]
e_full = [np.zeros_like(x_full[-1])]
Ze_full = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x_full[-1]]

szddpc.build_problem_simplified(2, 5, loss_callback, constraints_callback)

for t in range(total_steps):
    result, v, xbark, Zek = szddpc.solve(
        xbar_full[-1],
        e_full[-1],
        verbose=True
    )
    print(f'[{t}] x: {x_full[-1]} - xbar: {xbar_full[-1]} - v: {v[0]}')

    xbar_full.append(xbark[1])
    u = szddpc.theta.K @ x_full[-1] + v[0]
    x_next = sys.A @ x_full[-1] +  np.squeeze(sys.B @ u) + W.sample()
    x_full.append(x_next.flatten())
    e_full.append(x_full[-1] - xbar_full[-1])
    
    Zek = Zek.Z.value
    Ze_full.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar_full[-1])

x_full = np.array(x_full)
xbar_full = np.array(xbar_full)
e_full = np.array(e_full)


fig, ax = plt.subplots()
colors = cm.gray(np.linspace(0, 1, len(Ze_full)))

centers = []
for idx, Z in enumerate(Ze_full):
    centers.append(Z.center)
    collection = PatchCollection([Z.reduce(min(3, Z.order)).polygon],  facecolor='lightgray', edgecolor='black', lw=0.5)
    ax.add_collection(collection)
ax.set_xlim(-8, 0.5)
ax.set_ylim(-2.3, 2.4)

centers = np.array(centers)


plt.annotate('$x_0$',xy=(x_full[0,0]+0.05,x_full[0,1]+0.05),xytext=(x_full[0,:] + 0.5),
                arrowprops=dict(arrowstyle='-|>', fc="k", ec="k", lw=1.),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"))


plt.annotate(f'$x_{t+1}$',xy=(x_full[-1,0]-0.05,x_full[-1,1]-0.05),xytext=(x_full[-1,:] - 0.5),
                arrowprops=dict(arrowstyle='-|>', fc="k", ec="k", lw=1.),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"))
plt.plot(x_full[:,0], x_full[:,1], linestyle='dashdot', marker='x', color='black', linewidth=0.7)
plt.plot(centers[:,0], centers[:,1], linestyle='solid', color='black', linewidth=0.15)


plt.grid()
plt.show()
