"""naholos - Ukrainian word stress (наголос): dictionary + fast neural tagger."""
from .core import DEFAULT_MODEL, Naholos, stressify
from .text import ACUTE, apply_marks, marks_of, strip_marks, user_marks

__version__ = "0.1.0"
__all__ = ["Naholos", "stressify", "DEFAULT_MODEL", "ACUTE", "strip_marks", "marks_of",
           "apply_marks", "user_marks"]
