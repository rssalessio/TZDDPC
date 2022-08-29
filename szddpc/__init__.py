from .szddpc import SZDDPC
from .utils import (
    compute_theta,
    compute_A_B,
    compute_control_gain,
    spectral_radius
)
from .objects import (
    OptimizationProblemVariables,
    OptimizationProblem,
    Data,
    DataDrivenDataset,
    SystemZonotopes,
    Theta
)

__author__ = 'Alessio Russo - alessior@kth.se'
__version__ = '0.0.3'
__url__ = 'https://github.com/rssalessio/SZDPC'
__info__ = {
    'version': __version__,
    'author': __author__,
    'url': __url__
}