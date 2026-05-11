"""Package for spiking wavelet transforms and related tools."""

from ._version import __version__

from .wavelet import Wavelet
from .haar import HaarWavelet
from .morlet import MorletWavelet
from .szu import SzuWavelet
from .dog import DifferenceOfGaussiansWavelet
from .dot import DifferenceOfTimeCausalKernelsWavelet
from .doe import DifferenceOfExponentialsWavelet
from .spiking_dog import SpikingDoGWavelet
from .spiking_dot import SpikingDoTWavelet
from .spiking_doe import SpikingDoEWavelet

# Short aliases. The long names stay canonical for self-documenting code;
# the short names match the paper conventions and are friendlier to type.
DoG = DifferenceOfGaussiansWavelet
DoT = DifferenceOfTimeCausalKernelsWavelet
DoE = DifferenceOfExponentialsWavelet
SpikingDoG = SpikingDoGWavelet
SpikingDoT = SpikingDoTWavelet
SpikingDoE = SpikingDoEWavelet

__all__ = [
    "__version__",
    "Wavelet",
    "HaarWavelet",
    "MorletWavelet",
    "SzuWavelet",
    "DifferenceOfGaussiansWavelet",
    "DifferenceOfTimeCausalKernelsWavelet",
    "DifferenceOfExponentialsWavelet",
    "SpikingDoGWavelet",
    "SpikingDoTWavelet",
    "SpikingDoEWavelet",
    "DoG",
    "DoT",
    "DoE",
    "SpikingDoG",
    "SpikingDoT",
    "SpikingDoE",
]