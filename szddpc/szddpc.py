import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pyzonotope import MatrixZonotope, concatenate_zonotope, Zonotope, CVXZonotope, Interval
from pydatadrivenreachability import compute_LTI_matrix_zonotope
from szddpc.objects import OptimizationProblem, DataDrivenDataset, SystemZonotopes, Theta, Data
from szddpc.utils import compute_theta

class SZDDPC(object):
    optimization_problem: Union[OptimizationProblem,None] = None
    dataset: DataDrivenDataset
    zonotopes: SystemZonotopes
    Msigma: MatrixZonotope
    theta: Theta

    def __init__(self, data: Data):
        """
        Solves the ZPC optimization problem
        See also https://arxiv.org/pdf/2103.14110.pdf

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        self.update_identification_data(data)
    
    @property
    def num_samples(self) -> int:
        """ Return the number of samples used to estimate the Matrix Zonotope Msigma """
        return self.dataset.Um.shape[0] + 1

    @property
    def dim_u(self) -> int:
        """ Return the dimensionality of u (the control signal) """
        return self.dataset.Um.shape[1]

    @property
    def dim_x(self) -> int:
        """ Return the dimensionality of x (the state signal) """
        return self.dataset.Xp.shape[1]

    def update_identification_data(self, data: Data):
        """
        Update identification data matrices. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/state data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.x.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.x.shape[0] == data.u.shape[0], \
            "Input/state data must have the same length"

        Xm = data.x[:-1]
        Xp = data.x[1:]
        Um = data.u[:-1]

        self.dataset = DataDrivenDataset(Xp, Xm, Um, data)
        self.optimization_problem = None

    def build_zonotopes(self, zonotopes: SystemZonotopes):
        """
        [Private method] Do not invoke directly.
        Builds all the zonotopes needed to solve ZPC. 
        """
        X0, W, U, X = zonotopes.X0, zonotopes.W, zonotopes.U, zonotopes.X
        assert X0.dimension == W.dimension and X0.dimension == self.dim_x \
            and X.dimension == X0.dimension, \
            'The zonotopes do not have the correct dimension'
        
        print('--------------------------------------------')
        print('Building zonotopes')
        self.optimization_problem = None
        self.zonotopes = zonotopes
        Mw = concatenate_zonotope(W, self.num_samples - 1)

        self.Msigma = compute_LTI_matrix_zonotope(self.dataset.Xm, self.dataset.Xp, self.dataset.Um, Mw)
        print('--------------------------------------------')
        return self.Msigma

    def compute_theta(self, tol: float = 1e-5, num_max_iterations: int = 20, num_initial_points: int = 10) -> Theta:
        assert self.Msigma is not None, 'Msigma is not defined'
        # Simulate closed loop systems and gather trajectories        
        self.theta = compute_theta(self.Msigma, self.Msigma.center[:, :self.dim_x], self.Msigma.center[:, self.dim_x:],
            tol, num_initial_points, num_max_iterations)

        return self.theta

    def build_zonotopes_theta(self,
            zonotopes: SystemZonotopes,
            tol: float = 1e-5,
            num_max_iterations: int = 20,
            num_initial_points: int = 10
            ) -> Tuple[Theta, MatrixZonotope]:
        """
        Builds the ZPC optimization problem
        For more info check section 3.2 in https://arxiv.org/pdf/2103.14110.pdf

        :param zonotopes:           System zonotopes
        :param horizon:             Horizon length
        :param build_loss:          Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a scalar value of type Expression
        :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
                                    of shape (TxM), where T is the horizon length and M is the feature size
                                    The callback should return a list of constraints.
        :return:                    Parameters of the optimization problem
        """
        self.build_zonotopes(zonotopes)
        self.compute_theta(tol, num_max_iterations, num_initial_points)

        return self.theta, self.Msigma


    def solve(
            self,
            xbar0: np.ndarray,
            e0: np.ndarray,
            horizon: int,
            Zsigma: Zonotope,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            **cvxpy_kwargs
        ) -> Tuple[float, np.ndarray, np.ndarray, CVXZonotope]:
        """
        Solves the DeePC optimization problem
        For more info check alg. 2 in https://arxiv.org/pdf/1811.05890.pdf

        :param y0:                  The initial output
        :param cvxpy_kwargs:        All arguments that need to be passed to the cvxpy solve method.
        :return u_optimal:          Optimal input signal to be applied to the system, of length `horizon`
        :return info:               A dictionary with 5 keys:
                                    info['variables']: variables of the optimization problem
                                    info['value']: value of the optimization problem
                                    info['u_optimal']: the same as the first value returned by this function
        """
        assert build_loss is not None, "Loss function callback cannot be none"
        assert len(e0) == self.dim_x, f"Invalid size"

        # Build variables
        v = cp.Variable(shape=(horizon, self.dim_u))
        xbar = cp.Variable(shape=(horizon + 1, self.dim_x))
        ubar = cp.Variable(shape=(horizon, self.dim_u))

        # Acl = A+BK
        A, B = self.Msigma.center[:, :self.dim_x], self.Msigma.center[:, self.dim_x:]
        Acl = A + B @ self.theta.K

        #print(f'Max eig {np.abs(np.linalg.eig(Acl)[0]).max()}')
        
        beta_x = cp.Variable(shape=(horizon, self.zonotopes.X.num_generators))
        beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))

        constraints = [
            beta_u >= -1.,
            beta_u <= 1.,
            beta_x >= -1,
            beta_x <= 1,
            ubar == np.array([self.zonotopes.U.center] * horizon) + (beta_u @ self.zonotopes.U.generators.T),
            xbar[1:] == np.array([self.zonotopes.X.center] * horizon) + (beta_x @ self.zonotopes.X.generators.T),
            xbar[0] == xbar0,
            ubar == xbar[:-1] @ self.theta.K.T + v
        ]

        Ze: List[CVXZonotope] = [CVXZonotope(e0, np.zeros((self.dim_x, 1)))]

        term_1 = [self.zonotopes.W + self.zonotopes.sigma]
        term_2 = np.zeros(xbar[0].shape)

        for k in range(1,horizon):
            term_1.append(term_1[-1] * Acl + (self.zonotopes.W + self.zonotopes.sigma))

        for k in range(horizon):
            #print(f'Step {k}')
            Zx: Interval = (Ze[-1]+ xbar[k]).interval
            Zu: Interval = (Ze[-1] * self.theta.K + ubar[k]).interval
            constraints_k = [
                xbar[k+1] == A @ xbar[k] + B @ ubar[k],
                Zx.right_limit <= self.zonotopes.X.interval.right_limit,
                Zx.left_limit >= self.zonotopes.X.interval.left_limit,
                Zu.right_limit <=  self.zonotopes.U.interval.right_limit,
                Zu.left_limit >= self.zonotopes.U.interval.left_limit
            ]

            constraints.extend(constraints_k)

            ##
            #
            # Ze(t) = (A+BK)^t Ze(0)+ Sum_k=0^{t-1} (A+BK)^{k} (W+Sigma
            #
            #(Ze[-1] * Acl)# + self.zonotopes.W + self.zonotopes.sigma) + self.theta.deltaA @ xbar[k] + self.theta.deltaB @ ubar[k] 
            term_0 = Ze[0]*np.linalg.matrix_power(Acl, k+1)
            term_2 = term_2 @ Acl + self.theta.deltaA @ xbar[k] + self.theta.deltaB @ ubar[k] 

            Ze.append(
               term_0 + term_1[k] + term_2
            )

            #term_2 = term_2 * Acl
            

        _constraints = build_constraints(ubar, xbar[1:]) if build_constraints is not None else (None, None)
 
        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)
        
        # Build loss
        _loss = build_loss(ubar, xbar[1:])
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        problem_loss =_loss

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        assert problem.is_dcp(), 'Problem does not satisfy the DCP rules'
 

        try:
            result = problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            with open('zpc_logs.txt', 'w') as f:
                print(f'Error while solving the DeePC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        # print(f'Left: {Ze[1].interval.left_limit.value} - Right: {Ze[1].interval.right_limit.value} ')
        # print(term_1[-1].interval.right_limit)
        # print(term_2.value)
        return result, v.value, xbar.value, Ze[1]