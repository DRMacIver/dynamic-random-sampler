"""Dynamic Random Sampler - Python bindings for Rust implementation."""

from collections.abc import MutableMapping, MutableSequence

from dynamic_random_sampler.dynamic_random_sampler import (  # type: ignore[import-not-found]
    SamplerDict,
    SamplerList,
)

# Register with collections.abc abstract base classes
MutableSequence.register(SamplerList)
MutableMapping.register(SamplerDict)

__version__ = "0.1.0"
__all__ = ["SamplerDict", "SamplerList"]
