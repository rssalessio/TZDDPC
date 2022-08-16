import numpy as np
from scipy.signal import StateSpace
from pydatadrivenreachability import Zonotope
from typing import Tuple
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import cvxpy as cp
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
def estimated_autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = signal.correlate(x, x, mode = 'full')[-n:]
    #assert N.allclose(r, N.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

def generate_trajectories(
        sys: StateSpace,
        num_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a trajectory
    """

    dim_x, dim_u = sys.B.shape
    u = np.random.normal(size=(num_steps, dim_u))

    # Simulate system
    X = np.zeros((num_steps, dim_x))
    for i in range(1, num_steps):
            X[i, :] = sys.A @ X[i - 1, :] +  np.squeeze(sys.B * u[i - 1])

    return u, X

def solve_problem(sys: StateSpace, u: np.ndarray, x: np.ndarray):
    dim_x, dim_u = sys.B.shape
    T = x.shape[0]
    u_delta = cp.Variable(shape=(dim_u, T))

    x = x.reshape((dim_x, T))
    u = u.reshape((dim_u, T))
    A0 = np.zeros_like(sys.A)
    A0[0, 0 ] = 1

    obj = cp.Minimize(cp.norm(A0 @ x + sys.B @ u_delta))
    problem = cp.Problem(obj, [])
    res = problem.solve()
    return res, u_delta.value

def some_tests(sys: StateSpace, u: np.ndarray, x: np.ndarray):
    delta = np.random.normal(size=(u.shape[1], u.shape[0]-1))/ np.sqrt(2)
    Um = (u[:-1, :].T )/np.sqrt(2) + delta
    Xm = x[:-1, :].T
    Xp = x[1:].T

    AB = Xp @ np.linalg.pinv(np.vstack([Xm, Um]))
    print(AB)
    Xp = Xp - sys.A @ Xm - sys.B @ (Um)

    # K = Xm.T @ np.linalg.inv(Xm @ Xm.T)
    # K = np.vstack([Xm, Um])
    # K = K @ K.T
    # #print(f'EIG {np.linalg.svd(K)[1]}')
    # #print(K)
    # A0 = - sys.B @ delta @ Xm.T @ np.linalg.inv(Xm @ Xm.T)
    # print(A0)
    # alpha = np.linalg.inv(sys.B.T @ sys.B) @ sys.B.T @ A0 @ Xm @ delta.T @ np.linalg.inv(delta @ delta.T)
    # print(alpha)
    V = np.vstack([Xm, Um])
    M = np.linalg.inv(V @ V.T)
    print(np.linalg.svd(V @ V.T)[1])

    AB = Xp @ np.linalg.pinv(np.vstack([Xm, Um]))
    print(AB)


dt = 0.05
num = [0.28261, 0.50666]
den = [1, -1.41833, 1.58939, -1.31608, 0.88642]
sys = signal.TransferFunction(num, den, dt=dt).to_ss()
print(sys.A)
#print(sys.B)
u, x = generate_trajectories(sys, 10000)
print(some_tests(sys, u, x))


# res, u_delta = solve_problem(sys, u, x)
# # print(res)
# # print(u_delta)
# u_delta = u_delta.flatten()
# u2 = u_delta - u_delta.mean()
# u2 /= u2.std()
# plt.plot(u2.flatten())
# plt.show()
# print(np.std(u2))
# # plt.plot(estimated_autocorrelation(u_delta.flatten()))
# # plt.show()