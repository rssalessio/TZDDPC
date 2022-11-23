# To run this example you also need to install matplotlib
from plot_constants import *
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from tzddpc import TZDDPC, Data, SystemZonotopes
import tzddpc
from utils import generate_trajectories
from pyzonotope import Zonotope
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from pyzpc import ZPC, Data as DataZPC, SystemZonotopes as ZonotopesZPC
np.random.seed(25)


# Define the loss function
def loss_callback(u: cp.Variable, x: cp.Variable) -> Expression:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]

    cost = 0
    for i in range(horizon):
        cost += cp.norm(x[i,:], p=2) ** 2 + 1e-2 * cp.norm(u[i], p=1)
    return  cost

# Define additional constraints
def constraints_callback(u: cp.Variable, x: cp.Variable) -> List[Constraint]:
    horizon, dim_u, dim_x = u.shape[0], u.shape[1], x.shape[1]
    # Define a list of additional input/output constraints
    return [] #x[:,1] <= 2]

# Double integrator
A = np.array([[1, 1], [0, 1]])
B = np.array([[0.5],[1]])
dim_x, dim_u = B.shape
sys = scipysig.StateSpace(A, B, np.eye(dim_x), np.zeros((dim_x, dim_u)))

num_trajectories = 1
num_steps_per_trajectory = 100
total_steps = 12
horizon = min(10, total_steps)


# Define zonotopes and generate data
X0 = Zonotope([-5, -2], 0 * np.eye(dim_x))
U = Zonotope([0], 1*np.ones((1,1)))
W = Zonotope(np.zeros(dim_x), 0.1* np.array([[1, 0.5], [0.5, 1]]))
X = Zonotope([-4, 0], 0.95*np.diag([5,2.5]))
zonotopes = SystemZonotopes(X0, U, X, W)

W_vertices = W.compute_vertices()
num_W_vertices = len(W_vertices)
print(f'Vertices of Zw: {W_vertices}')

data = generate_trajectories(sys, X0, U, W, num_trajectories, num_steps_per_trajectory)
x0 = X0.sample().flatten()


# Build TZDDPC
tzddpc = TZDDPC(data)
tzddpc.build_zonotopes_theta(zonotopes)


x = [x0]
xbar = [x[-1].copy()]
e = [np.zeros_like(x[-1])]
Ze = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]

tzddpc.build_problem(2, loss_callback, constraints_callback)

for t in range(total_steps):
    result, v, xbark, Zek = tzddpc.solve(
        xbar[-1],
        e[-1],
        verbose=False
    )
    print(f'[{t}] x: {x[-1]} - xbar: {xbar[-1]} - v: {v[0]}')

    xbar.append(xbark[1])
    u = tzddpc.theta.K @ e[-1] + v[0]
    x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) +  W_vertices[np.random.choice(num_W_vertices)]
    x.append(x_next.flatten())
    e.append(x[-1] - xbar[-1])
    
    Zek = Zek.Z.value
    Ze.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar[-1])

x_full = np.array(x)
xbar_full = np.array(xbar)
e_full = np.array(e)
Ze_full = Ze

# x = [x0]
# xbar = [x[-1].copy()]
# e = [np.zeros_like(x[-1])]
# Ze = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]

# tzddpc.build_problem_simplified(2, 3, loss_callback, constraints_callback)

# for t in range(total_steps):
#     result, v, xbark, Zek = tzddpc.solve(
#         xbar[-1],
#         e[-1],
#         verbose=False
#     )
#     print(f'[{t}] x: {x[-1]} - xbar: {xbar[-1]} - v: {v[0]}')

#     xbar.append(xbark[1])
#     u = tzddpc.theta.K @ e[-1] + v[0]
#     x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W_vertices[np.random.choice(num_W_vertices)]
#     x.append(x_next.flatten())
#     e.append(x[-1] - xbar[-1])
    
#     Zek = Zek.Z.value
#     Ze.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar[-1])

x_simplified = np.array(x)
xbar_simplified = np.array(xbar)
e_simplified = np.array(e)
Ze_simplified = Ze


x = [x0]
for t in range(total_steps):

    xbar.append(xbark[1])
    u = tzddpc.theta.K @ x[-1]
    x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) + W_vertices[np.random.choice(num_W_vertices)]
    x.append(x_next.flatten())

x_lqr = np.array(x)


# BUILD ZPC
print('BUILDING ZPC')
X = Zonotope([-4, 0], 1.5*np.diag([5,2.5]))
zpc_zonotopes = ZonotopesZPC(X0, U, X, W, Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))), Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))))
zpc = ZPC(DataZPC(data.u, data.x))

problem = zpc.build_problem(zpc_zonotopes, 2, loss_callback, constraints_callback)
x = [x0]
Ze_zpc = [Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))) + x[-1]]
for n in range(total_steps):
    print(f'Solving step {n}')

    result, info = zpc.solve(x[-1], verbose=False,warm_start=True)
    #print(problem.variables.y0.value)
    #print(problem.variables.s.value)
    # import pdb
    # pdb.set_trace()

    u = info['u_optimal']
    x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u[0]) + W_vertices[np.random.choice(num_W_vertices)]
    x.append(x_next.flatten())
    Zval = info['reachable_set'][1].Z.value
    Ze_zpc.append(Zonotope(Zval[:,0], Zval[:, 1:]))
    #Ze.append(Zonotope(Zek[:, 0], Zek[:, 1:]) + xbar[-1])

x_zpc = np.array(x)

fig, ax = plt.subplots(figsize=(12,6))

centers = []
for idx, Z in enumerate(Ze_full):
    centers.append(Z.center)
    collection = PatchCollection([Z.reduce(min(3, Z.order)).polygon],  facecolor='lightgray', edgecolor='black', lw=0.5)
    ax.add_collection(collection)

    Z = Ze_zpc[idx]
    collection = PatchCollection([Z.reduce(min(3, Z.order)).polygon],  facecolor='lightsalmon', edgecolor='black', lw=0.5)
    ax.add_collection(collection)
ax.set_xlim(-9.2, 1.2)
ax.set_ylim(-2.7, 2.7)

centers = np.array(centers)

d = np.linspace(-10,10,300)
xd,yd = np.meshgrid(d,d)
plt.imshow( ((xd<=-9*0.95) | (yd<=-2.5*0.95) |  (xd>=1*0.95)  | (yd>=2.5*0.95)).astype(int) , 
                extent=(xd.min(),xd.max(),yd.min(),yd.max()),origin="lower", cmap="Greys", alpha = 0.4)


plt.annotate('$t=0$',xy=(x_full[0,0]+0.05,x_full[0,1]+0.05),xytext=(x_full[0,:] + 0.5),
                arrowprops=dict(arrowstyle='-|>', fc="k", ec="k", lw=1.),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"), fontsize=20)


plt.annotate(f'$t={t+1}$',xy=(x_simplified[-2,0]-0.05,x_simplified[-2,1]-0.05),xytext=(x_simplified[-2,:] - 0.8),
                arrowprops=dict(arrowstyle='-|>', fc="k", ec="k", lw=1.),
                bbox=dict(pad=0, facecolor="none", edgecolor="none"), fontsize=20)
line1, = plt.plot(x_full[:,0], x_full[:,1], linestyle='solid', marker='x', color='black', linewidth=0.7,label='TZ-DDPC - $x_t$')
# line2, = plt.plot(x_simplified[:,0], x_simplified[:,1], linestyle='dashed', marker='o', color='black', linewidth=0.7,label='Simplified TZDDPC - $x_t$')
line3, = plt.plot(x_zpc[:,0], x_zpc[:,1], linestyle='dotted', marker='v', color='red', linewidth=0.7,label='ZPC -  $x_t$')
#plt.plot(centers[:,0], centers[:,1], linestyle='solid', color='black', linewidth=0.15)
plt.xlabel('$x_1$', horizontalalignment='right', x=.95)
plt.ylabel('$x_2$', horizontalalignment='right', y=.95)
plt.legend(fancybox = True, facecolor="whitesmoke", 
            handles = [
                Patch(color='lightgray', label='TZ-DDPC - $\\bar{\\mathcal{Z}}_{e,t}$'),
                line1,
                Patch(color='lightsalmon', label='ZPC -  predicted reachable set'),
                #line2,
                line3
            ], loc='lower right')


#fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.grid()
plt.savefig('figures/double_integrator.pdf',bbox_inches='tight')


plt.show()
