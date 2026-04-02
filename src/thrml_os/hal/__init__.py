"""Hardware Abstraction Layer for THRML-OS."""

from thrml_os.hal.backend import Backend, BackendType
from thrml_os.hal.jax_backend import JAXBackend
from thrml_os.hal.simulator import SimulatorBackend
from thrml_os.hal.registry import BackendRegistry, get_default_backend

__all__ = [
    "Backend",
    "BackendType",
    "JAXBackend",
    "SimulatorBackend",
    "BackendRegistry",
    "get_default_backend",
]
