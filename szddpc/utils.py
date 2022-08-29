import numpy as np
from typing import Tuple
from pyzonotope import MatrixZonotope
from szddpc.objects import Theta
import cvxpy as cp

def spectral_radius(X: np.ndarray) -> float:
    """ Returns the spectral radius of a matrix """
    assert len(X.shape) == 2 and X.shape[0] == X.shape[1], 'X is not  a square matrix'
    return np.abs(np.linalg.eigvals(X)).max()

def compute_A_B(Mdata: MatrixZonotope, K: np.ndarray, num_init: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """ Computes the adversarial matrices (A,B) for a given control gain K and set of matrices M """
    dim_u, dim_x = K.shape
    A = cp.Variable((dim_x, dim_x))
    B = cp.Variable((dim_x, dim_u))

    beta_A = cp.Variable(Mdata.num_generators)
    beta_B = cp.Variable(Mdata.num_generators)

    constraints = [
        beta_A >= -1, beta_A <= 1,
        beta_B >= -1, beta_B <= 1
    ]

    Agen = Mdata.center[:, :dim_x]
    Bgen = Mdata.center[:, dim_x:]
    for i in range(Mdata.num_generators):
        Agen = Agen + Mdata.generators[i][:, :dim_x] * beta_A[i]
        Bgen = Bgen + Mdata.generators[i][:, dim_x:] * beta_B[i]

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

def compute_theta(
        Mdata: MatrixZonotope,
        A0: np.ndarray,
        B0: np.ndarray,
        tolerance: float = 1e-5,
        initial_points: int = 10,
        max_iterations: int = 20,
        accuracy: float = 1e-2,
        confidence: float = 1e-5) -> Theta:
    assert Mdata.contains(np.hstack([A0,B0])), 'Mdata does not contain (A0,B0)'
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
        
        An, Bn = compute_A_B(Mdata, Kn, initial_points)
        
        lambda_max = max(spectral_radius(An + Bn @ Kn), spectral_radius(A0+B0@Kn))
        print(f'[Iteration {iteration}] Closed loop spectral radius: {lambda_init}->{lambda_max} - Adversarial spectral radius: {lambda_adv} - K {Kn.flatten()}')
        if np.abs(lambda_max - prev_lambda_max) < tolerance or lambda_max < 1:
            break
        
        iteration += 1
        prev_lambda_max = lambda_max

    print(f'Radius {np.abs(np.linalg.eig(A0+B0 @ Kn)[0])}')
    print(f'Optimization completed. Closed loop spectral radius: {lambda_max} - K {Kn.flatten()}')
    print('--------------------------------------------')

    assert Mdata.contains(np.hstack([An,Bn])), 'M does not contain (An,Bn)'
    assert is_gain_robust(Mdata, Kn, accuracy, confidence), f'K is not robust with accuracy-confidence of {accuracy,1-confidence}'

    return Theta(Kn, An - A0, Bn - B0)

def is_gain_robust(Mdata: MatrixZonotope, K: np.ndarray, accuracy: float, confidence: float) -> bool:
    """
    Verifies that a certain gain K is robust for given (accuracy, confidence) levels. Robustness is guaranteed
    w.r.t. a uniform distribution over Mdata.

    Returns True if the gain is robust
    """
    assert K.shape[1] == Mdata.shape[0], 'Wrong dimensionality for K'
    assert accuracy > 0 and accuracy < 1, 'Accuracy should be in (0,1)'
    assert confidence > 0 and confidence < 1, 'confidence should be in (0,1)'

    dim_n = K.shape[1]

    N = np.log(1/confidence) / np.log(1 / (1 - accuracy))

    for n in range(N):
        X = Mdata.sample()
        A, B = X[:, :dim_n], X[:, dim_n:]

        if spectral_radius(A + B @K ) >= 1.:
            return False
    
    return True