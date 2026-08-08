"""Backward-compatible VeinCast model import.

New code should import :class:`VeinCast` from ``veincast``. The historical
``AllWeatherGraphQueryModel`` name is retained so older scripts can load the
same state dictionaries without renaming parameter keys.
"""

from veincast import VeinCast


AllWeatherGraphQueryModel = VeinCast

__all__ = ["VeinCast", "AllWeatherGraphQueryModel"]
