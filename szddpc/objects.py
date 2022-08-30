import numpy as np
from typing import NamedTuple, List
from cvxpy import Expression, Variable, Problem, Parameter
from cvxpy.constraints.constraint import Constraint
from pyzonotope import Zonotope

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
    """
    X0: Zonotope
    U: Zonotope
    X: Zonotope
    W: Zonotope

class Theta(NamedTuple):
    K: np.ndarray
    deltaA: np.ndarray
    deltaB: np.ndarray
