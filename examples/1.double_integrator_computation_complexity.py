# To run this example you also need to install matplotlib
from plot_constants import *
import numpy as np
import scipy.signal as scipysig
import cvxpy as cp
from typing import List
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from tzddpc import TZDDPC, Data, SystemZonotopes
import tzddpc
from utils import generate_trajectories
from pyzonotope import Zonotope
from pyzpc import ZPC, Data as DataZPC, SystemZonotopes as ZonotopesZPC
from argparse import ArgumentParser
import time
import os, psutil
import multiprocessing as mp
import threading
import pickle
np.random.seed(25)

k0 = 1
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

# Constraints enlargement
C_TZDDPC = 1.2
C_ZPC = 1.2

# Define zonotopes and generate data
X0 = Zonotope([-5, -2], 0 * np.eye(dim_x))
U = Zonotope([0], 1*np.ones((1,1)))
W = Zonotope(np.zeros(dim_x), 0.001* np.array([[1, 0.5], [0.5, 1]]))
X = Zonotope([-4, 0], C_TZDDPC*np.diag([5,2.5]))
zonotopes = SystemZonotopes(X0, U, X, W)

W_vertices = W.compute_vertices()
num_W_vertices = len(W_vertices)
print(f'Vertices of Zw: {W_vertices}')

data = generate_trajectories(sys, X0, U, W, num_trajectories, num_steps_per_trajectory)


def simulateTZDDPC(x0: np.ndarray, horizon):
    # Build TZDDPC
    tzddpc = TZDDPC(data)
    tzddpc.build_zonotopes_theta(zonotopes)
    x = [x0]
    xbar = [x[-1].copy()]
    e = [np.zeros_like(x[-1])]

    tzddpc.build_problem(horizon, loss_callback, constraints_callback)

    for t in range(1):
        result, v, xbark, Zek = tzddpc.solve(
            xbar[-1],
            e[-1],
            verbose=False
        )
        xbar.append(xbark[1])
        u = tzddpc.theta.K @ e[-1] + v[0]
        x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) +  W_vertices[np.random.choice(num_W_vertices)]
        x.append(x_next.flatten())
        e.append(x[-1] - xbar[-1])
        


def simulateZPC(x0, horizon):
    # BUILD ZPC
    X = Zonotope([-4, 0], C_ZPC*np.diag([5,2.5]))
    zpc_zonotopes = ZonotopesZPC(X0, U, X, W, Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))), Zonotope(np.zeros(dim_x), np.zeros((dim_x,1))))
    zpc = ZPC(DataZPC(data.u, data.x))

    problem = zpc.build_problem(zpc_zonotopes, horizon, loss_callback, constraints_callback)
    x = [x0]
    for n in range(1):
        result, info = zpc.solve(x[-1], verbose=False,warm_start=True)
        u = info['u_optimal']
        x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u[0]) + W_vertices[np.random.choice(num_W_vertices)]
        x.append(x_next.flatten())

def simulateSimplifiedTZDDPC(x0, horizon):
    # Build TZDDPC
    tzddpc = TZDDPC(data)
    tzddpc.build_zonotopes_theta(zonotopes)
    x = [x0]
    xbar = [x[-1].copy()]
    e = [np.zeros_like(x[-1])]

    tzddpc.build_problem_simplified(1, horizon, loss_callback, constraints_callback)

    for t in range(1):
        result, v, xbark, Zek = tzddpc.solve(
            xbar[-1],
            e[-1],
            verbose=False
        )
        xbar.append(xbark[1])
        u = tzddpc.theta.K @ e[-1] + v[0]
        x_next = sys.A @ x[-1] +  np.squeeze(sys.B @ u) +  W_vertices[np.random.choice(num_W_vertices)]
        x.append(x_next.flatten())
        e.append(x[-1] - xbar[-1])


def process_memory(process: psutil.Process) -> float:
    memory = process.memory_info().rss
    child_memory = 0
    for child in process.children(recursive=True):
        child_memory += child.memory_info().rss
    return child_memory

def evaluate_memory(process: psutil.Process, thread_event: threading.Event, memory_usage: List[float], PERIOD: int = 10):
    CMEM = 1024 * 1024
    memory_usage.append(process_memory(process)/ CMEM)
    while(not thread_event.is_set()):
        time.sleep(PERIOD)
        memory_usage.append(process_memory(process)/ CMEM)
        print(f'{memory_usage[-1]} - {memory_usage[-1]/(1+memory_usage[0])}')
        


def evaluate_method(fun, max_horizon = 10 , N_EVALS = 10, PERIOD_EVAL_MEMORY: int = 5):
    process = psutil.Process(os.getpid())
    execution_times = {horizon: [] for horizon in range(1, max_horizon+1)}
    memory_usage = {horizon: [] for horizon in range(1, max_horizon+1)}

    
    x0 = [X0.sample().flatten() for i in range(N_EVALS)]
    for horizon in range(1, max_horizon + 1):
        thread_event = threading.Event()
        thread = threading.Thread(target = evaluate_memory, args = (process, thread_event,  memory_usage[horizon], PERIOD_EVAL_MEMORY))
        
        
        thread.start()
        for i in range(N_EVALS):
            p = mp.Process(target=fun, args=(x0[i], horizon))
            p.start()
            start = time.time()
            p.join(40 * 60)
            p.terminate()
            p.kill()
            #fun(x0[i], horizon)
            execution_times[horizon].append(time.time() - start)
        
        thread_event.set()
        thread.join()


    for i in range(1, max_horizon+1):
        execution_times[i] = np.array(execution_times[i])
        memory_usage[i] = np.array(memory_usage[i]) 
        memory_usage[i] = memory_usage[i][1:] - memory_usage[i][0]
    return {'time': execution_times, 'memory': memory_usage}

#04:26
if __name__ == '__main__':
    PERIOD_EVAL_MEMORY = 1
    parser = ArgumentParser()
    parser.add_argument("-m", "--method", dest="method",
                        help="method to run [ZPC|TZDDPC|STZDDPC]", default='TZDDPC')
    parser.add_argument("-n", "--nevals", dest="n_evals",
                        help="number of simulations", default=5, type=int)
    parser.add_argument("-ho", "--horizon", dest="horizon",
                        help="horizon", default=3, type=int)
        
    parser.add_argument("-k0", "--k0", dest="k0",
                        help="k0 value", default=1, type=int)
    args = parser.parse_args()
    method: str = args.method.upper()
    results = []

    k0 = args.k0
    if (method == 'ZPC'):
        results = evaluate_method(simulateZPC,  args.horizon, args.n_evals, PERIOD_EVAL_MEMORY)
    elif (method == 'TZDDPC'):
        results = evaluate_method(simulateTZDDPC,  args.horizon, args.n_evals, PERIOD_EVAL_MEMORY)
    elif (method == 'STZDDPC'):
        results = evaluate_method(simulateSimplifiedTZDDPC,  args.horizon, args.n_evals, PERIOD_EVAL_MEMORY)
    else:
        raise ValueError(f'Could not find method {args.method}')
    
    with open(f'data_{method}.pkl', 'wb') as f:
        pickle.dump(results, f, protocol = pickle.HIGHEST_PROTOCOL)

