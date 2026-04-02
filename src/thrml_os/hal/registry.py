"""Backend registry for managing available compute backends."""

from __future__ import annotations

from typing import Dict, Optional, Type

from thrml_os.hal.backend import Backend, BackendType


class BackendRegistry:
    """Registry of available compute backends.
    
    Manages backend discovery, instantiation, and selection.
    """
    
    _backends: Dict[BackendType, Type[Backend]] = {}
    _instances: Dict[str, Backend] = {}
    _default_backend: Optional[Backend] = None
    
    @classmethod
    def register(cls, backend_type: BackendType, backend_class: Type[Backend]) -> None:
        """Register a backend class.
        
        Args:
            backend_type: Type identifier for the backend
            backend_class: Backend class to register
        """
        cls._backends[backend_type] = backend_class
    
    @classmethod
    def get(cls, backend_type: BackendType, **kwargs) -> Backend:
        """Get or create a backend instance.
        
        Args:
            backend_type: Type of backend to get
            **kwargs: Arguments to pass to backend constructor
            
        Returns:
            Backend instance
        """
        # Create unique key for this configuration
        key = f"{backend_type.value}:{hash(frozenset(kwargs.items()))}"
        
        if key not in cls._instances:
            if backend_type not in cls._backends:
                raise ValueError(f"Unknown backend type: {backend_type}")
            
            backend_class = cls._backends[backend_type]
            instance = backend_class(**kwargs)
            instance.initialize()
            cls._instances[key] = instance
        
        return cls._instances[key]
    
    @classmethod
    def list_available(cls) -> list[BackendType]:
        """List all registered backend types.
        
        Returns:
            List of available backend types
        """
        return list(cls._backends.keys())
    
    @classmethod
    def set_default(cls, backend: Backend) -> None:
        """Set the default backend.
        
        Args:
            backend: Backend to use as default
        """
        cls._default_backend = backend
    
    @classmethod
    def get_default(cls) -> Optional[Backend]:
        """Get the default backend.
        
        Returns:
            Default backend or None
        """
        return cls._default_backend
    
    @classmethod
    def shutdown_all(cls) -> None:
        """Shutdown all backend instances."""
        for instance in cls._instances.values():
            try:
                instance.shutdown()
            except Exception:
                pass
        cls._instances.clear()
        cls._default_backend = None


def get_default_backend() -> Backend:
    """Get the default backend, auto-detecting if needed.
    
    Returns:
        Best available backend for the current system
    """
    # Check if already set
    default = BackendRegistry.get_default()
    if default is not None:
        return default
    
    # Auto-detect best backend
    from thrml_os.hal.jax_backend import JAXBackend
    from thrml_os.hal.simulator import SimulatorBackend
    
    # Register backends if not already done
    BackendRegistry.register(BackendType.JAX_CPU, JAXBackend)
    BackendRegistry.register(BackendType.JAX_GPU, JAXBackend)
    BackendRegistry.register(BackendType.JAX_TPU, JAXBackend)
    BackendRegistry.register(BackendType.SIMULATOR, SimulatorBackend)
    
    # Try to detect GPU
    try:
        import jax
        devices = jax.devices()
        for dev in devices:
            if dev.platform == "gpu":
                backend = JAXBackend(device="gpu")
                backend.initialize()
                BackendRegistry.set_default(backend)
                return backend
            elif dev.platform == "tpu":
                backend = JAXBackend(device="tpu")
                backend.initialize()
                BackendRegistry.set_default(backend)
                return backend
    except Exception:
        pass
    
    # Fall back to CPU
    try:
        backend = JAXBackend(device="cpu")
        backend.initialize()
        BackendRegistry.set_default(backend)
        return backend
    except Exception:
        pass
    
    # Last resort: simulator
    backend = SimulatorBackend()
    backend.initialize()
    BackendRegistry.set_default(backend)
    return backend
