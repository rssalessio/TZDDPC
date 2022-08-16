from itertools import combinations
from typing import Type, NamedTuple
import numpy as np
import cvxpy as cp
from pydatadrivenreachability import Zonotope

class System(NamedTuple):
    a: Zonotope
    b: Zonotope
    w: Zonotope

a = Zonotope([0], np.diag([2]))
w = Zonotope([0], 0.2 * np.diag([1]))
b = Zonotope([1], np.diag([1]))
sys = System(a, b, w)

def scenario_approach_solve_optimization_problem(sys: Type[System], horizon: int, num_scenarios: int):
    u = cp.Variable(shape=(horizon))
    x = cp.Variable(shape=(num_scenarios,horizon + 1))
    gamma = cp.Variable(nonneg=True)
    losses = []

    constraints = [-1 <= u, u <= 1]
    x0 = 0
    for i in range(num_scenarios):
        constraints.append(x[i,0] == x0)
        sysA = sys.a.sample()[0]
        sysB = sys.b.sample()[0]
        W = sys.w.sample(batch_size=horizon)

        loss = 0

        for t in range(horizon):
            constraints.append(x[i, t+1] == sysA * x[i, t] + sysB * u[t] + W[t].flatten())
            loss = loss + x[i,t] ** 2 + u[t] ** 2

 
        constraints.append(loss <= gamma)
        losses.append(loss)
    
    objective = cp.Minimize(gamma)
    problem = cp.Problem(objective, constraints)
    try:
        res = problem.solve(verbose=False, solver=cp.MOSEK)
    except Exception as e:
        raise Exception(f'Could not solve problem, error {e}')

    return res, constraints, losses


def vertex_solve_optimization_problem(sys: Type[System], horizon: int, num_scenarios: int):
    
    vertex_a = sys.a.compute_vertices()
    vertex_b = sys.b.compute_vertices()
    vertex_w = sys.w.compute_vertices()

    num_a, num_b, num_w = vertex_a.shape[0], vertex_b.shape[0], vertex_w.shape[0]
    combinations = (num_a*num_b*num_w)

    u = cp.Variable(shape=(horizon))
    x = cp.Variable(shape=(horizon + 1))
    gamma = cp.Variable(nonneg=True)
    losses = []


    constraints = [-1 <= u, u <= 1]
    x0 = 0




    for i in range(num_scenarios):
        constraints.append(x[i,0] == x0)
        sysA = sys.a.sample()[0]
        sysB = sys.b.sample()[0]
        W = sys.w.sample(batch_size=horizon)

        loss = 0

        for t in range(horizon):
            constraints.append(x[i, t+1] == sysA * x[i, t] + sysB * u[t] + W[t].flatten())
            loss = loss + x[i,t] ** 2 + u[t] ** 2

 
        constraints.append(loss <= gamma)
        losses.append(loss)
    
    objective = cp.Minimize(gamma)
    problem = cp.Problem(objective, constraints)
    try:
        res = problem.solve(verbose=False, solver=cp.MOSEK)
    except Exception as e:
        raise Exception(f'Could not solve problem, error {e}')

    return res, constraints, losses

print(a.compute_vertices())
print(b.compute_vertices())
print(w.compute_vertices())

for i in [1,5,10,20,30,50,100]:
    res, constraints, losses = scenario_approach_solve_optimization_problem(sys, 4, i)
    print(f'i {i} res {res}')#- {[x.value for x in losses]}')



