import numpy as np
from typing import NamedTuple, Tuple, Optional, List, Union
from cvxpy import Expression, Variable, Problem, Parameter
from cvxpy.constraints.constraint import Constraint
from pydatadrivenreachability import Zonotope, MatrixZonotope
import cvxpy as cp
import dccp

class OptimizationProblemVariables(NamedTuple):
    """
    Class used to store all the variables used in the optimization
    problem
    """
    y0: Parameter
    u: Variable
    y: Variable
    s_l: Variable
    s_u: Variable
    beta_u: Variable

class OptimizationProblem(NamedTuple):
    """
    Class used to store the elements an optimization problem
    :param problem_variables:   variables of the opt. problem
    :param constraints:         constraints of the problem
    :param objective_function:  objective function
    :param problem:             optimization problem object
    """
    variables: OptimizationProblemVariables
    constraints: List[Constraint]
    objective_function: Expression
    problem: Problem


class Data(NamedTuple):
    """
    Tuple that contains input/output data
    :param u: input data
    :param x: state data
    """
    u: np.ndarray
    x: np.ndarray


class DataDrivenDataset(NamedTuple):
    """
    Tuple that contains input/output data splitted
    according to X+, X- and U- (resp. Yp, Ym, Um).
    See also section 2.3 in https://arxiv.org/pdf/2103.14110.pdf
    """
    Xp: np.ndarray
    Xm: np.ndarray
    Um: np.ndarray
    original_data: Data


class SystemZonotopes(NamedTuple):
    """
    Tuple that contains the zonotopes of the system

    :param X0: initial condition zonotope
    :param U: Input zonotope
    :param X: Output zonotope
    :param W: process noise zonotope
    :param sigma: Disturbance zonotope
    """
    X0: Zonotope
    U: Zonotope
    X: Zonotope
    W: Zonotope
    sigma: Zonotope

class Theta(NamedTuple):
    K: np.ndarray
    deltaA: np.ndarray
    deltaB: np.ndarray

def spectral_radius(X: np.ndarray) -> float:
    """ Returns the spectral radius of a matrix """
    assert len(X.shape) == 2 and X.shape[0] == X.shape[1], 'X is not  a square matrix'
    return np.abs(np.linalg.eigvals(X)).max()


def compute_maximization(M: MatrixZonotope, K: np.ndarray) -> np.ndarray:
    M = M.reduce(1).choose_columns([0, 1, 2, 3, 4])
    np_shape = np.prod(M.shape)
    G = np.zeros((np_shape, M.num_generators))
    for i in range(M.num_generators):
        G[:, i] += M.generators[i].flatten()
    
    c = M.center.flatten()

    omega = cp.Variable((np_shape))
    y  = cp.Variable(M.num_generators, nonneg=True)
    s = cp.Variable(M.num_generators, nonneg=True)
    t = cp.Variable(M.num_generators, nonneg=True)
    b = cp.Variable(M.num_generators, boolean=True)

    import pdb
    pdb.set_trace()
    obj = cp.sum(y) + c.T @ omega - cp.norm(omega, p=2)
    l1_term = G.T @ omega
    constraints = []
    for i in range(M.num_generators):
        constraints.extend([
            cp.norm(l1_term, p=1) <= cp.sum(y),
            y[i] == l1_term[i] + s[i],
            y[i] == -l1_term[i] + t[i],
            s[i] <= (1e12) * b[i],
            t[i] <= (1e12) * (1 - b[i])
        ])
    problem = cp.Problem(cp.Maximize(obj), constraints)
    res = problem.solve(verbose=True, solver=cp.MOSEK)
    print(f'Result: {res} - {omega.value}')

    beta = cp.Variable(M.num_generators)
    problem = cp.Problem(cp.Maximize((G.T @ omega.value).T @ beta), [beta >= -1, beta <= 1])
    res = problem.solve()
    print(f'Result {res} {beta.value}')
    F =M.center
    for i in range(M.num_generators):
        F += M.generators[i] * beta.value[i]
    print(f'{F} {np.abs(np.linalg.eig(F)[0])}')

    dim_u, dim_x = K.shape
    A = cp.Variable((dim_x, dim_x))

    beta_A = cp.Variable(M.num_generators)

    constraints = [
        beta_A >= -1, beta_A <= 1,
    ]

    Agen = M.center[:, :dim_x]
    for i in range(M.num_generators):
        Agen = Agen + M.generators[i][:, :dim_x] * beta_A[i]

    constraints.append(A == Agen)
    problem = cp.Problem(cp.Maximize(cp.norm(A, p='fro')), constraints)

    res = problem.solve(method='dccp', solver=cp.MOSEK, verbose=False, ccp_times=50)
    print(f'DCCP res {res[0]} - {A.value} - {np.abs(np.linalg.eig(A.value)[0])}')
    import pdb
    pdb.set_trace()
    X = np.reshape(omega.value, M.shape)
    print(np.linalg.eig(X)[0])
    import pdb
    pdb.set_trace()

    print(X)



def compute_A_B(Msigma: MatrixZonotope, K: np.ndarray, num_init: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the adversarial matrices (A,B) for a given control gain K and set of matrices M """
    dim_u, dim_x = K.shape
    A = cp.Variable((dim_x, dim_x))
    B = cp.Variable((dim_x, dim_u))

    beta_A = cp.Variable(Msigma.num_generators)
    beta_B = cp.Variable(Msigma.num_generators)

    constraints = [
        beta_A >= -1, beta_A <= 1,
        beta_B >= -1, beta_B <= 1
    ]

    Agen = Msigma.center[:, :dim_x]
    Bgen = Msigma.center[:, dim_x:]
    for i in range(Msigma.num_generators):
        Agen = Agen + Msigma.generators[i][:, :dim_x] * beta_A[i]
        Bgen = Bgen + Msigma.generators[i][:, dim_x:] * beta_B[i]

    constraints.append(A == Agen)
    constraints.append(B == Bgen)
    problem = cp.Problem(cp.Maximize(cp.norm(A + B @ K, p='fro')), constraints)

    res = problem.solve(method='dccp', solver=cp.MOSEK, verbose=False, ccp_times=num_init)

    An = A.value
    Bn = B.value
    return An, Bn

def compute_control_gain(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """ Compute stabilizing matrix K given a pair (A,B) """
    # Compute K
    dim_x, dim_u = B.shape
    Z = cp.Variable((dim_u, dim_x))
    X = cp.Variable((dim_x, dim_x), symmetric=True)

    Fn = A @ X + B @ Z
    P = cp.bmat([[X, Fn], [Fn.T, X]])
    constraints = [X >> 0, P >> 0]

    problem = cp.Problem(cp.Minimize(1), constraints)
    res = problem.solve()

    # Return K
    return Z.value @ np.linalg.inv(X.value)

def compute_theta(Msigma: MatrixZonotope, A0: np.ndarray, B0: np.ndarray, tolerance: float = 1e-5, initial_points: int = 10, max_iterations: int = 20) -> Theta:
    assert Msigma.contains(np.hstack([A0,B0])), 'M does not contain (A0,B0)'
    dim_x, dim_u = B0.shape

    An = A0.copy()
    Bn = B0.copy()
    Kn = np.zeros((dim_u, dim_x))
    prev_lambda_max = 0
    iteration = 0
    print('--------------------------------------------')
    print(f'Computing theta=(K,\Delta A,\Delta B) - Initial spectral radius: {spectral_radius(An + Bn @ Kn)}')
    while iteration < max_iterations:
        lambda_init = spectral_radius(An + Bn @ Kn)
        Kn = compute_control_gain(An, Bn)
 
        lambda_adv = spectral_radius(An + Bn @ Kn)
        
        An, Bn = compute_A_B(Msigma, Kn, initial_points)
        
        lambda_max = max(spectral_radius(An + Bn @ Kn), spectral_radius(A0+B0@Kn))
        print(f'[Iteration {iteration}] Closed loop spectral radius: {lambda_init}->{lambda_max} - Adversarial spectral radius: {lambda_adv} - K {Kn.flatten()}')
        if np.abs(lambda_max - prev_lambda_max) < tolerance or lambda_max < 1:
            break
        
        iteration += 1
        prev_lambda_max = lambda_max

    print(f'Radius {np.abs(np.linalg.eig(A0+B0 @ Kn)[0])}')
    print(f'Optimization completed. Closed loop spectral radius: {lambda_max} - K {Kn.flatten()}')
    print('--------------------------------------------')

    assert Msigma.contains(np.hstack([An,Bn])), 'M does not contain (An,Bn)'
    return Theta(Kn, An - A0, Bn - B0)

def compute_theta2(M: MatrixZonotope, A0: np.ndarray, B0: np.ndarray, tolerance: float = 1e-5, max_iterations: int = 20) -> Theta:
    assert M.contains(np.hstack([A0,B0])), 'M does not contain (A0,B0)'

    dim_x, dim_u = B0.shape

    An = A0
    Bn = B0
    Kn = np.zeros((dim_u, dim_x))
    prev_lambda_max = 0
    iteration = 0
    print('--------------------------------------------')
    print(f'Computing theta=(K,\Delta A,\Delta B) - Initial spectral radius: {spectral_radius(An + Bn @ Kn)}')
    while iteration < max_iterations:
        lambda_init = spectral_radius(An + Bn @ Kn)
        Kn = compute_control_gain(An, Bn)
        lambda_adv = spectral_radius(An + Bn @ Kn)
        
        An, Bn = compute_A_B(M, Kn)
        
        
        
        lambda_max = spectral_radius(An + Bn @ Kn)
        print(f'[Iteration {iteration}] Closed loop spectral radius: {lambda_init}->{lambda_max} - Adversarial spectral radius: {lambda_adv} - K {Kn.flatten()}')
        if np.abs(lambda_max - prev_lambda_max) < tolerance:
            break
        
        iteration += 1
        prev_lambda_max = lambda_max

    print(f'Optimization completed. Closed loop spectral radius: {lambda_max} - K {Kn.flatten()}')
    print('--------------------------------------------')
    return Theta(Kn, An - A0, Bn - B0)

def compute_theta3(M: MatrixZonotope, A0: np.ndarray, B0: np.ndarray, tolerance: float = 1e-5, max_iterations: int = 20) -> Theta:
    assert M.contains(np.hstack([A0,B0])), 'M does not contain (A0,B0)'

    dim_x, dim_u = B0.shape

    An = A0
    Bn = B0
    Kn = np.zeros((dim_u, dim_x))
    prev_lambda_max = 0
    iteration = 0
    print('--------------------------------------------')
    print(f'Computing theta=(K,\Delta A,\Delta B) - Initial spectral radius: {spectral_radius(An + Bn @ Kn)}')
    while iteration < max_iterations:
        lambda_init = spectral_radius(An + Bn @ Kn)
        K = compute_control_gain(An, Bn)
        An, Bn = compute_A_B(M, Kn)
        
        
        lambda_adv = spectral_radius(An + Bn @ Kn)
        Kn = K
        
        lambda_max = spectral_radius(An + Bn @ Kn)
        print(f'[Iteration {iteration}] Closed loop spectral radius: {lambda_init}->{lambda_max} - Adversarial spectral radius: {lambda_adv} - K {Kn.flatten()}')
        if np.abs(lambda_max - prev_lambda_max) < tolerance:
            break
        
        iteration += 1
        prev_lambda_max = lambda_max

    print(f'Optimization completed. Closed loop spectral radius: {lambda_max} - K {Kn.flatten()}')
    print('--------------------------------------------')
    return Theta(Kn, An - A0, Bn - B0)