from cmath import isinf
import numpy as np
import cvxpy as cp
from typing import Tuple, Callable, List, Optional, Union, Dict
from cvxpy.expressions.expression import Expression
from cvxpy.constraints.constraint import Constraint
from pydatadrivenreachability import (
    concatenate_zonotope,
    compute_IO_LTI_matrix_zonotope,
    MatrixZonotope,
    Zonotope,
    CVXZonotope)

from szddpc.utils import (
    Data,
    DataDrivenDataset,
    SystemZonotopes,
    OptimizationProblem,
    OptimizationProblemVariables)
import sys
#sys.setrecursionlimit(10000)


class SZDDPC(object):
    optimization_problem: Union[OptimizationProblem,None] = None
    dataset: DataDrivenDataset
    zonotopes: SystemZonotopes
    Msigma: MatrixZonotope

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
    def dim_y(self) -> int:
        """ Return the dimensionality of y (the output signal) """
        return self.dataset.Yp.shape[1]

    def update_identification_data(self, data: Data):
        """
        Update identification data matrices of ZPC. You need to rebuild the optimization problem
        after calling this funciton.

        :param data:                A tuple of input/output data. Data should have shape TxM
                                    where T is the batch size and M is the number of features
        """
        assert len(data.u.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert len(data.y.shape) == 2, \
            "Data needs to be shaped as a TxM matrix (T is the number of samples and M is the number of features)"
        assert data.y.shape[0] == data.u.shape[0], \
            "Input/output data must have the same length"

        Ym = data.y[:-1]
        Yp = data.y[1:]
        Um = data.u[:-1]
        self.dataset = DataDrivenDataset(Yp, Ym, Um)
        self.optimization_problem = None

    def _build_zonotopes(self, zonotopes: SystemZonotopes):
        """
        [Private method] Do not invoke directly.
        Builds all the zonotopes needed to solve ZPC. 
        """
        X0, W, V, Av, U, Y = zonotopes.X0, zonotopes.W, zonotopes.V, zonotopes.Av, zonotopes.U, zonotopes.Y
        assert X0.dimension == W.dimension and X0.dimension == self.dim_y \
            and V.dimension == W.dimension and Av.dimension == V.dimension and Y.dimension == X0.dimension, \
            'The zonotopes do not have the correct dimension'
        
        self.zonotopes = zonotopes
        Mw = concatenate_zonotope(W, self.num_samples - 1)
        Mv = concatenate_zonotope(V, self.num_samples - 1)
        Mav = concatenate_zonotope(Av, self.num_samples - 1)

        self.Msigma = compute_IO_LTI_matrix_zonotope(self.dataset.Ym, self.dataset.Yp, self.dataset.Um, Mw, Mv, Mav)


    def build_problem(self,
            zonotopes: SystemZonotopes,
            horizon: int,
            build_loss: Callable[[cp.Variable, cp.Variable], Expression],
            build_constraints: Optional[Callable[[cp.Variable, cp.Variable], Optional[List[Constraint]]]] = None) -> OptimizationProblem:
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
        self._build_zonotopes(zonotopes)

        Z = self.zonotopes.W + self.zonotopes.V + (-1 *self.zonotopes.Av)

        # Build variables
        num_trajectories = 10
        y0 = cp.Parameter(shape=(self.dim_y))
        u = cp.Variable(shape=(horizon, self.dim_u))
        y = [cp.Variable(shape=(horizon + 1, self.dim_y)) for x in range(num_trajectories)]
        znoise = [cp.Variable(shape=(horizon, self.dim_y)) for x in range(num_trajectories)]
        beta_u = cp.Variable(shape=(horizon, self.zonotopes.U.num_generators))
        beta_z = [cp.Variable(shape=(horizon, Z.num_generators)) for x in range(num_trajectories)]
        beta_y = [cp.Variable(shape=(horizon, self.zonotopes.Y.num_generators)) for x in range(num_trajectories)]
        gamma = cp.Variable(nonneg=True)
        #import pdb
        #pdb.set_trace()
        constraints = [
            beta_u >= -1.,
            beta_u <= 1.,
            u == np.array([self.zonotopes.U.center] * horizon) + (beta_u @ self.zonotopes.U.generators.T),
        ]

        leftY = self.zonotopes.Y.interval.left_limit
        rightY = self.zonotopes.Y.interval.right_limit

        R = []

        for k in range(num_trajectories):
            constraints.extend([
                y[k][1:] <= np.array([rightY.flatten()] * horizon),
                y[k][1:] >= np.array([leftY.flatten()] * horizon),
                znoise[k] == np.array([Z.center] * horizon) + (beta_z[k] @ Z.generators.T),
                beta_z[k]  >= -1.,
                beta_z[k]  <= 1.,
                y[k][0] == y0
            ])

            sys_sample: np.ndarray = self.Msigma.sample()[0]
            sys_A = sys_sample[:, :-self.dim_u]
            sys_B = sys_sample[:, -self.dim_u:]

            #R.append([CVXZonotope(y0, np.zeros((self.dim_y, 1)))])


            for i in range(horizon):
                print(f'{k}-{i}')
                #R_ki = R[k][i]
                #import pdb
                #pdb.set_trace()
                # Rnew: CVXZonotope = R[k][i] * sys_A + CVXZonotope(u[i], np.zeros((self.dim_u, 1))) * sys_B +  Z
                # R[k].append(Rnew)
                # Rinterval = Rnew.interval
                constraints.append(y[k][i+1] == sys_A @ y[k][i] + sys_B @ u[i] + Z.sample()[0])#  znoise[k][i])
                # constraints.extend([
                #     Rinterval.right_limit <= rightY,
                #     Rinterval.left_limit >= leftY
                # ])


            _constraints = build_constraints(u, y[k]) if build_constraints is not None else (None, None)

            for idx, constraint in enumerate(_constraints):
                if constraint is None or not isinstance(constraint, Constraint) or not constraint.is_dcp():
                    raise Exception(f'Constraint {idx} is not defined or is not convex.')

            constraints.extend([] if _constraints is None else _constraints)

            # Build loss
            _loss = build_loss(u, y[k])
            
            if _loss is None or not isinstance(_loss, Expression) or not _loss.is_dcp():
                raise Exception('Loss function is not defined or is not convex!')

            constraints.append(_loss <= gamma)

        problem_loss = gamma

        # Solve problem
        objective = cp.Minimize(problem_loss)

        try:
            problem = cp.Problem(objective, constraints)
        except cp.SolverError as e:
            raise Exception(f'Error while constructing the DeePC problem. Details: {e}')

        self.optimization_problem = OptimizationProblem(
            variables = OptimizationProblemVariables(y0=y0, u=u, y=y, s_l=beta_z, s_u=gamma, beta_u=beta_u),
            constraints = constraints,
            objective_function = problem_loss,
            problem = problem
        )

        return self.optimization_problem


    def solve(
            self,
            y0: np.ndarray,
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
        assert len(y0) == self.dim_y, f"Invalid size"
        assert self.optimization_problem is not None, "Problem was not built"


        self.optimization_problem.variables.y0.value = y0
        try:
            #import pdb
            #pdb.set_trace()
            result = self.optimization_problem.problem.solve(**cvxpy_kwargs)
        except cp.SolverError as e:
            with open('zpc_logs.txt', 'w') as f:
                print(f'Error while solving the DeePC problem. Details: {e}', file=f)
            raise Exception(f'Error while solving the DeePC problem. Details: {e}')

        if np.isinf(result):
            import pdb
            pdb.set_trace()
            print(self.optimization_problem.problem.parameters)
            raise Exception('Problem is unbounded')

        print(self.optimization_problem.variables.s_l[1].value)

        u_optimal = self.optimization_problem.variables.u.value
        info = {
            'value': result, 
            'variables': self.optimization_problem.variables,
            'u_optimal': u_optimal
            }

        return u_optimal, info