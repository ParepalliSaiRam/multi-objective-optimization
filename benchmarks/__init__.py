"""Optimization benchmark suites used by this project."""

from .cec2022 import CEC2022_BENCHMARKS, cec2022, make_cec2022
from .cec2020_mmo import CEC2020_MMO_PROBLEMS, evaluate_mmo, make_mmo

__all__ = ["CEC2022_BENCHMARKS", "cec2022", "make_cec2022", "CEC2020_MMO_PROBLEMS", "evaluate_mmo", "make_mmo"]
