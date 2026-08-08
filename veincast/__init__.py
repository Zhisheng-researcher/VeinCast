"""VeinCast global medium-range weather forecasting package."""

from .model import VeinCast
from .variables import VariableRegistry

__all__ = ["VeinCast", "VariableRegistry"]
__version__ = "0.1.0"
