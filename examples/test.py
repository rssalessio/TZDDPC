import cvxpy as cp
from pydatadrivenreachability import Zonotope, CVXZonotope
import numpy as np
dim_x = 4



W = Zonotope([1] * dim_x, np.ones((dim_x,2)))
x= W.sample()[0]
dim_g = W.num_generators
beta = cp.Variable((dim_g))
#gamma = cp.Variable(nonneg=True)

constraints = [beta >= -1., beta <= 1.]
objective = cp.Minimize( cp.norm(x - (W.center + W.generators @ beta),p=2))
problem = cp.Problem(objective, constraints)
res = problem.solve()

print(beta.value)
print(res)
print(f'{x} vs {W.center + W.generators @ beta.value}')