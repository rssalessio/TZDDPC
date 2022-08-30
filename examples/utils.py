import numpy as np
from scipy.signal import StateSpace
from pyzonotope import Zonotope
from szddpc import Data

def generate_trajectories(
        sys: StateSpace,
        X0: Zonotope,
        U: Zonotope,
        W: Zonotope,     
        num_trajectories: int,
        num_steps: int) -> Data:
    """
    Generates trajectories from the system. We assume full state
    measurement (i.e., C=i)

    X0,U,W are respectively the zonotopes of the: 
        - initial condition
        - control signal
        - Process noise
    
    Returns a data object
    """

    dim_x, dim_u = sys.B.shape
    total_samples = num_steps * num_trajectories
    u = U.sample(total_samples).reshape((num_trajectories, num_steps, dim_u))

    # Simulate system
    X = np.zeros((num_trajectories, num_steps, dim_x))
    Y = np.zeros((num_trajectories, num_steps, dim_x))
    for j in range(num_trajectories):
        X[j, 0, :] = X0.sample()
        for i in range(1, num_steps):
            X[j, i, :] = sys.A @ X[j, i - 1, :] +  np.squeeze(sys.B * u[j, i - 1]) + W.sample()

            # We assume C = I
            Y[j, i, :] = X[j, i, :]
    
    y = np.reshape(Y, (num_steps * num_trajectories, dim_x))
    u = np.reshape(u, (num_steps * num_trajectories, dim_u))

    return Data(u, y)
