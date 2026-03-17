"""
JIT (Just-In-Time) compiler API, similar to torch.jit.
"""

from .api import trace
from .api import TracedModule, TracedFunction

__all__ = [
    "trace",
    "TracedModule",
    "TracedFunction",
]
