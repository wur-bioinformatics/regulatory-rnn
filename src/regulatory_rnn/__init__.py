from importlib.metadata import version

__version__ = version(__name__)

from .model import RegulatoryRNN
from .datasets import AdaptPulseDataset, GapGeneDataset
