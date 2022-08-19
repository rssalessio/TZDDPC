from cmath import isinf
import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydatadrivenreachability import (
    concatenate_zonotope,
    compute_LTI_matrix_zonotope,
    MatrixZonotope,
    Zonotope,
    CVXZonotope)

from szddpc.utils import (
    Data,
    Theta,
    compute_theta,
    DataDrivenDataset,
    SystemZonotopes,
    OptimizationProblem,
    OptimizationProblemVariables)
import sys
from scipy.linalg import solve_discrete_are
#sys.setrecursionlimit(10000)


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
        self.dataset = DataDrivenDataset(Xp, Xm, Um)
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
        self.theta = compute_theta(self.Msigma, self.Msigma.center[:, :self.dim_x], self.Msigma.center[:, self.dim_x:],
            tol, num_initial_points, num_max_iterations)
        return self.theta


    def build_problem(self,
            zonotopes: SystemZonotopes,
            horizon: int,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None,
            tol: float = 1e-5,
            num_max_iterations: int = 20,
            num_initial_points: int = 10
            ) -> OptimizationProblem:
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
        assert build_loss is not None, "Loss function callback cannot be none"

        self.optimization_problem = None
        self.build_zonotopes(zonotopes)
        self.compute_theta(tol, num_max_iterations, num_initial_points)

        # Build variables
        e0 = cp.Parameter(shape=(self.dim_x))
        v = cp.Variable(shape=(horizon, self.dim_u))
        xbar = cp.Variable(shape=(horizon + 1, self.dim_x))
        sigma = cp.Parameter(shape=(self.dim_x))
        ubar = cp.Variable(shape=(horizon, self.dim_u))
        xbar0 = cp.Parameter(shape=(self.dim_x))

        # Acl = A+BK
        A, B = self.Msigma.center[:, :self.dim_x], self.Msigma.center[:, self.dim_x:]
        Acl = A + B @ self.theta.K

        
        beta_x = cp.Variable(shape=(horizon, self.zonotopes.X.num_generators))
        beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))

        #import pdb
        #pdb.set_trace()
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

        import pdb
        pdb.set_trace()
        for k in range(horizon):
            print(f'Step {k}')
            interval = Ze[-1].interval
            Zx = (Ze[-1]+ xbar).interval
            Zu = (Ze[-1] * self.theta.K + ubar).interval
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


        import pdb
        pdb.set_trace()
        _constraints = build_constraints(ubar, xbar) if build_constraints is not None else (None, None)
 
        for idx, constraint in enumerate(_constraints):
            if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                raise Exception(f'Constraint {idx} is not defined or is not convex.')

        constraints.extend([] if _constraints is None else _constraints)
        
        # Build loss
        _loss = build_loss(ubar, xbar)
        
        if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
            raise Exception('Loss function is not defined or is not convex!')

        # import pdb
        # pdb.set_trace()
        problem_loss =_loss

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        import pdb
        pdb.set_trace()
        C = [x.is_dcp() for x in constraints]
        print(f'DCP : {problem.is_dcp()}')
        # self.optimization_problem = OptimizationProblem(
        #     variables = OptimizationProblemVariables(y0=y0, u=u, y=y, s_l=beta_z, s_u=gamma, beta_u=beta_u),
        #     constraints = constraints,
        #     objective_function = problem_loss,
        #     problem = problem
        # )

        # return self.optimization_problem


    # def build_problem2(self,
    #         zonotopes: SystemZonotopes,
    #         horizon: int,
    #         build_loss: Callable[[cp.Variable, cp.Variable], Expression],
    #         build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None) -> OptimizationProblem:
    #     """
    #     Builds the ZPC optimization problem
    #     For more info check section 3.2 in https://arxiv.org/pdf/2103.14110.pdf

    #     :param zonotopes:           System zonotopes
    #     :param horizon:             Horizon length
    #     :param build_loss:          Callback function that takes as input an (input,output) tuple of data
    #                                 of shape (TxM), where T is the horizon length and M is the feature size
    #                                 The callback should return a scalar value of type Expression
    #     :param build_constraints:   Callback function that takes as input an (input,output) tuple of data
    #                                 of shape (TxM), where T is the horizon length and M is the feature size
    #                                 The callback should return a list of constraints.
    #     :return:                    Parameters of the optimization problem
    #     """
    #     assert build_loss is not None, "Loss function callback cannot be none"

    #     self.optimization_problem = None
    #     self._build_zonotopes(zonotopes)

    #     Z: Zonotope = self.zonotopes.W + self.zonotopes.V + (-1 *self.zonotopes.Av)

    #     # Build variables
    #     num_trajectories = 5
    #     y0 = cp.Parameter(shape=(self.dim_y))
    #     u = cp.Variable(shape=(horizon, self.dim_u))
    #     y = [cp.Variable(shape=(horizon + 1, self.dim_y)) for x in range(num_trajectories)]
    #     znoise = [cp.Variable(shape=(horizon, self.dim_y)) for x in range(num_trajectories)]
    #     beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))
    #     beta_z = [cp.Variable(shape=(horizon, Z.num_generators)) for x in range(num_trajectories)]
    #     beta_y = [cp.Variable(shape=(horizon, self.zonotopes.Y.num_generators)) for x in range(num_trajectories)]
    #     gamma = cp.Variable(nonneg=True)
    #     rho = [cp.Variable(shape=(horizon, self.dim_y),nonneg=True) for x in range(num_trajectories)]
    #     P = cp.Variable((self.dim_y, self.dim_y), PSD=True)
    #     x0  = cp.Variable((self.dim_y))

    #     #import pdb
    #     #pdb.set_trace()
    #     constraints = [
    #         beta_u >= -1.,
    #         beta_u <= 1.,
    #         u == np.array([self.zonotopes.U.center] * horizon) + (beta_u @ self.zonotopes.U.generators.T),
    #     ]

    #     leftY = self.zonotopes.Y.interval.left_limit
    #     rightY = self.zonotopes.Y.interval.right_limit

    #     R = []


    #     for k in range(num_trajectories):
    #         constraints.extend([
    #             #x0 <= rightY.flatten(),# * horizon),
    #             #x0 >= leftY.flatten(),# * horizon),
    #             znoise[k] == np.array([Z.center] * horizon) + (beta_z[k] @ Z.generators.T),
    #             beta_z[k]  >= -1.,
    #             beta_z[k]  <= 1.,
    #             y[k][0] == y0
    #         ])



    #         sys_sample: np.ndarray = self.Msigma.sample()[0]
    #         sys_A = sys_sample[:, :-self.dim_u]
    #         sys_B = sys_sample[:, -self.dim_u:]

    #         #R.append([CVXZonotope(y0, np.zeros((self.dim_y, 1)))])

    #         S = Z

    #         Psys = solve_discrete_are(sys_A, sys_B, np.eye(sys_A.shape[0]), np.zeros(sys_B.shape[1]))
    #         Ksys = -np.linalg.inv(sys_B.T @ Psys @ sys_B) @ sys_B.T @ Psys @ sys_A

    #         Apow = np.eye(sys_A.shape[0])

    #         for i in range(horizon):
    #             print(f'{k}-{i}')
    #             #R_ki = R[k][i]
    #             #import pdb
    #             #pdb.set_trace()
    #             # Rnew: CVXZonotope = R[k][i] * sys_A + CVXZonotope(u[i], np.zeros((self.dim_u, 1))) * sys_B +  Z
    #             # R[k].append(Rnew)
    #             # Rinterval = Rnew.interval
    #             constraints.append(y[k][i+1] == sys_A @ y[k][i] + sys_B @ u[i])# + Z.sample()[0])#  znoise[k][i])
    #             # constraints.extend([
    #             #     Rinterval.right_limit <= rightY,
    #             #     Rinterval.left_limit >= leftY
    #             # ])
    #             constraints.append(y[k][i] + S.interval.right_limit <= rightY +rho[k][i])
    #             constraints.append(y[k][i] - S.interval.left_limit >= leftY - rho[k][i])
    #             # import pdb
    #             # pdb.set_trace()
    #             Apow = Apow @ sys_A #+ sys_B @ Ksys)
    #             S = S + Z *Apow
    #             eigs = np.abs(np.linalg.eig(Apow)[0]).max()
    #             print(f'{i} - {S.interval.left_limit} - {S.interval.right_limit} - {eigs}')
    #             #Si = y[k][i+1] + Z.interval.right_limit

    #         _constraints = build_constraints(u, y[k]) if build_constraints is not None else (None, None)
    #         #import pdb
    #         #pdb.set_trace()
    #         for idx, constraint in enumerate(_constraints):
    #             if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
    #                 raise Exception(f'Constraint {idx} is not defined or is not convex.')

    #         constraints.extend([] if _constraints is None else _constraints)
            
    #         # Build loss
    #         _loss = build_loss(u, y[k])
            
    #         if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
    #             raise Exception('Loss function is not defined or is not convex!')

    #         constraints.append(_loss <= gamma)
    #     # import pdb
    #     # pdb.set_trace()
    #     problem_loss = gamma
    #     for i in range(num_trajectories):
    #         problem_loss += cp.sum(cp.norm(rho[k], axis=1))

    #     # Solve problem
    #     objective = cp.Minimize(problem_loss)

    #     try:
    #         problem = cp.Problem(objective, constraints)
    #     except cp.SolverError as e:
    #         raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

    #     self.optimization_problem = OptimizationProblem(
    #         variables = OptimizationProblemVariables(y0=y0, u=u, y=y, s_l=beta_z, s_u=gamma, beta_u=beta_u),
    #         constraints = constraints,
    #         objective_function = problem_loss,
    #         problem = problem
    #     )

    #     return self.optimization_problem


    def solve(
            self,
            e0: np.ndarray,
            **cvxpy_kwargs
        ) -> Tuple[np.ndarray, Dict[str, Union[float, np.ndarray, OptimizationProblemVariables]]]:
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
        assert len(e0) == self.dim_x, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        #self.optimization_problem.variables.y0.value = y0
        try:
            #import pdb
            #pdb.set_trace()
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            with open('zpc_logs.txt', 'w') as f:
                print(f'Error while solving the DeePC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            # import pdb
            # pdb.set_trace()
            print(self.optimization_problem.problem.parameters)
            raise Exception('Problem is unbounded')

        #print(self.optimization_problem.variables.s_l[1].value)

        u_optimal = self.optimization_problem.variables.u.value
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info