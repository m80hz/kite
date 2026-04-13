"""qa package exports.

Provides the VLM adapter and a simple factory.
"""
from .adapter import RoboFACAdapter  # noqa: F401


def get_adapter(model_name: str):
	"""Return appropriate adapter instance given model name."""
	return RoboFACAdapter()

__all__ = [
	"RoboFACAdapter",
	"get_adapter",
]
