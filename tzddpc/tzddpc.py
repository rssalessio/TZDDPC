import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pyzonotope import MatrixZonotope, concatenate_zonotope, Zonotope, CVXZonotope, Interval
from pydatadrivenreachability import compute_LTI_matrix_zonotope
from tzddpc.objects import OptimizationProblem, DataDrivenDataset, SystemZonotopes, Theta, Data
from tzddpc.utils import compute_theta

class TZDDPC(object):
    optimization_problem: Union[OptimizationProblem,None] = None
    dataset: DataDrivenDataset
    zonotopes: SystemZonotopes
    Mdata: MatrixZonotope
    Mdelta: MatrixZonotope
    MdataK: MatrixZonotope
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
        """ Return the number of samples used to estimate the Matrix Zonotope Mdata """
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

        self.Mdata = compute_LTI_matrix_zonotope(self.dataset.Xm, self.dataset.Xp, self.dataset.Um, Mw)
        print('--------------------------------------------')
        return self.Mdata

    def compute_theta(self, tol: float = 1e-5, num_max_iterations: int = 20, num_initial_points: int = 10) -> Theta:
        assert self.Mdata is not None, 'Mdata is not defined'
        # Simulate closed loop systems and gather trajectories        
        self.theta = compute_theta(self.Mdata, self.Mdata.center[:, :self.dim_x], self.Mdata.center[:, self.dim_x:],
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

        # Create Mdelta,K
        self.MdataK = self.Mdata * np.vstack([np.eye(self.dim_x), self.theta.K])

        # Create MDelta
        delta = np.hstack([self.Mdata.center[:, :self.dim_x], self.Mdata.center[:, self.dim_x:]])
        self.Mdelta: MatrixZonotope = self.Mdata + (-1 * delta)

        # Reduce order
        self.Mdata = self.Mdata.reduce(1)
        self.MdataK = self.MdataK.reduce(1)
        self.Mdelta = self.Mdelta.reduce(1)

        return self.theta, self.Mdata

    def build_problem(
            self,
            horizon: int,
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
        #assert len(e0) == self.dim_x, f"Invalid size"

        # Build variables
        e0 = cp.Parameter(shape=(self.dim_x))
        v = cp.Variable(shape=(horizon, self.dim_u))
        xbar0 = cp.Parameter(shape=(self.dim_x))
        xbar = cp.Variable(shape=(horizon + 1, self.dim_x))
        x = cp.Variable(shape=(horizon, self.dim_x))
        # ubar = cp.Variable(shape=(horizon, self.dim_u))

        # Acl = A+BK
        A, B = self.Mdata.center[:, :self.dim_x], self.Mdata.center[:, self.dim_x:]
        #Acl = A + B @ self.theta.K

        constraints = [
            xbar[0] == xbar0,
            # ubar == xbar[:-1] @ self.theta.K.T + v,
            xbar[1:] ==  xbar[:-1] @ A.T + v @ B.T
        ]

        Ze: List[CVXZonotope] = [CVXZonotope(e0, np.zeros((self.dim_x, 1)))]

        XU = [CVXZonotope(cp.hstack((xbar[k], v[k])), np.zeros((self.dim_x + self.dim_u, 1))) for k in range(horizon)]
        Ze_new_term1 = [self.MdataK * Ze[0] ]
        Z_noise = [self.Mdelta * XU[k] + self.zonotopes.W for k in range(horizon)]
        Ze_new_term2 = []

    
        for k in range(horizon):
            Ze_new_term1.append(self.MdataK * Ze_new_term1[-1])

            noise_term =  Z_noise[0]
            for j in range(1, k):
                noise_term = self.MdataK * noise_term + Z_noise[j]
            Ze_new_term2.append(noise_term)


        for k in range(horizon):
            print(f'Step {k}')
            Zx: Interval = (Ze[-1]+ xbar[k]).interval
            Zu: Interval = (Ze[-1] * self.theta.K + v[k]).interval
            constraints_k = [
                Zx.right_limit <= self.zonotopes.X.interval.right_limit,
                Zx.left_limit >= self.zonotopes.X.interval.left_limit,
                Zu.right_limit <=  self.zonotopes.U.interval.right_limit,
                Zu.left_limit >= self.zonotopes.U.interval.left_limit,
                x[k] == (Ze[-1]+ xbar[k]).center
            ]

            # Ze_new_term1 = self.MdataK * Ze[0] 
            # for i in range(k):
            #     Ze_new_term1 = self.MdataK * Ze_new_term1
            # Ze_new_term2 = self.Mdelta * CVXZonotope(cp.hstack((xbar[k], ubar[k])), np.zeros((self.dim_x + self.dim_u, 1)))
            Ze_new: CVXZonotope =Ze_new_term1[k] + Ze_new_term2[k]# Ze_new_term1 + Ze_new_term2 + self.zonotopes.W
            print(Ze_new.num_generators)
            Ze.append(Ze_new)

            constraints.extend(constraints_k)

            
        _constraints = build_constraints(v, xbar) if build_constraints is not None else (None, None)
 
        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)
        
        # Build loss
        _loss = build_loss(v, x)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        problem_loss =_loss

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            self.problem_full = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the TZDDPC problem. Details: {e}')
        assert self.problem_full.is_dcp(), 'Problem does not satisfy the DCP rules'


        self.parameters = (e0, xbar0)
        self.variables = (v, xbar, Ze)
        return self.problem_full

    def build_problem_simplified(
            self,
            k0: int,
            horizon: int,
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
        #assert len(e0) == self.dim_x, f"Invalid size"

        # Build variables
        e0 = cp.Parameter(shape=(self.dim_x))
        v = cp.Variable(shape=(horizon, self.dim_u))
        xbar0 = cp.Parameter(shape=(self.dim_x))
        xbar = cp.Variable(shape=(horizon + 1, self.dim_x))
        #ubar = cp.Variable(shape=(horizon, self.dim_u))

        # Acl = A+BK
        A, B = self.Mdata.center[:, :self.dim_x], self.Mdata.center[:, self.dim_x:]
        #Acl = A + B @ self.theta.K

        constraints = [
            xbar[0] == xbar0,
            #ubar == xbar[:-1] @ self.theta.K.T + v,
            xbar[1:] ==  xbar[:-1] @ A.T + v @ B.T
        ]

        Ze: List[CVXZonotope] = [CVXZonotope(e0, np.zeros((self.dim_x, 1)))]

        XU = [CVXZonotope(cp.hstack((xbar[k], v[k])), np.zeros((self.dim_x + self.dim_u, 1))) for k in range(horizon)]
        Ze_new_term1 = [self.MdataK * Ze[0] ]
        Z_noise = [self.Mdelta * XU[k] + self.zonotopes.W for k in range(horizon)]
        Ze_new_term2 = []

    
        for k in range(horizon):
            if k > k0:
                Ze_new_term1.append(Ze_new_term1[-1])
            else:
                Ze_new_term1.append(self.MdataK * Ze_new_term1[-1])

            start_idx = max(0, k-k0)
            noise_term =  Z_noise[start_idx]
            for j in range(1, min(k, k0)):
                noise_term = self.MdataK * noise_term + Z_noise[start_idx + j]

            Ze_new_term2.append(noise_term)


        for k in range(horizon):
            print(f'Step {k}')
            Zx: Interval = (Ze[-1]+ xbar[k]).interval
            Zu: Interval = (Ze[-1] * self.theta.K + v[k]).interval
            constraints_k = [
                Zx.right_limit <= self.zonotopes.X.interval.right_limit,
                Zx.left_limit >= self.zonotopes.X.interval.left_limit,
                Zu.right_limit <=  self.zonotopes.U.interval.right_limit,
                Zu.left_limit >= self.zonotopes.U.interval.left_limit
            ]

            # Ze_new_term1 = self.MdataK * Ze[0] 
            # for i in range(k):
            #     Ze_new_term1 = self.MdataK * Ze_new_term1
            # Ze_new_term2 = self.Mdelta * CVXZonotope(cp.hstack((xbar[k], ubar[k])), np.zeros((self.dim_x + self.dim_u, 1)))
            Ze_new: CVXZonotope =Ze_new_term1[k] + Ze_new_term2[k]# Ze_new_term1 + Ze_new_term2 + self.zonotopes.W
            print(Ze_new.num_generators)
            Ze.append(Ze_new)

            constraints.extend(constraints_k)

            
        _constraints = build_constraints(v, xbar[1:]) if build_constraints is not None else (None, None)
 
        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)
        
        # Build loss
        _loss = build_loss(v, xbar[1:])
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        problem_loss =_loss

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            self.problem_full = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the TZDDPC problem. Details: {e}')
        assert self.problem_full.is_dcp(), 'Problem does not satisfy the DCP rules'


        self.parameters = (e0, xbar0)
        self.variables = (v, xbar, Ze)
        return self.problem_full

    def solve(
            self,
            xbar0: np.ndarray,
            e0: np.ndarray,
            **cvxpy_kwargs
        ) -> Tuple[float, np.ndarray, np.ndarray, CVXZonotope]:
 
        self.parameters[0].value = e0
        self.parameters[1].value = xbar0
        try:
            result = self.problem_full.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            with open('zpc_logs.txt', 'w') as f:
                print(f'Error while solving the TZDDPC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the TZDDPC problem. Details: {e}')


        if np.isinf(result):
            raise Exception('Problem is unbounded')

        return result, self.variables[0].value, self.variables[1].value, self.variables[2][1]

    

    def solve_simplified2(
            self,
            xbar0: np.ndarray,
            e0: np.ndarray,
            horizon: int,
            Zsigma: List[Zonotope],
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
        assert len(Zsigma) == horizon, "Zsigma needs to be a list of zonotopes of length == N, the horizon"

        # Build variables
        v = cp.Variable(shape=(horizon, self.dim_u))
        xbar = cp.Variable(shape=(horizon + 1, self.dim_x))
        ubar = cp.Variable(shape=(horizon, self.dim_u))

        # Acl = A+BK
        A, B = self.Mdata.center[:, :self.dim_x], self.Mdata.center[:, self.dim_x:]
        Acl = A + B @ self.theta.K
        
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

        term_1 = [self.zonotopes.W + Zsigma[0]]
        term_2 = np.zeros(self.dim_x)

        for k in range(1,horizon):
            term_1.append(term_1[-1] * Acl + (self.zonotopes.W + Zsigma[k]))

        for k in range(horizon):
            # print(f'[Simplified] Step {k}')
            Zx: Interval = (Ze[-1]+ xbar[k]).interval
            Zu: Interval = (Ze[-1] * self.theta.K + ubar[k]).interval
            constraints_k = [
                xbar[k+1] == Acl @ xbar[k] + B @ v[k],
                Zx.right_limit <= self.zonotopes.X.interval.right_limit,
                Zx.left_limit >= self.zonotopes.X.interval.left_limit,
                Zu.right_limit <=  self.zonotopes.U.interval.right_limit,
                Zu.left_limit >= self.zonotopes.U.interval.left_limit
            ]

            constraints.extend(constraints_k)

            # Ze(t) = (A+BK)^t Ze(0)+ Sum_k=0^{t-1} (A+BK)^{k} (W+Sigma
            term_0 = Ze[0]*np.linalg.matrix_power(Acl, k+1)
            term_2 = term_2 @ Acl + self.theta.deltaA @ xbar[k] + self.theta.deltaB @ ubar[k] 

            Ze.append(
               term_0 + term_1[k] + term_2
            )

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
            with open('tzddpc.txt', 'w') as f:
                print(f'Error while solving the simplified TZDDPC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the simplified TZDDPC problem. Details: {e}')

        if np.isinf(result):
            raise Exception('Problem is unbounded')

        return result, v.value, xbar.value, Ze[1]