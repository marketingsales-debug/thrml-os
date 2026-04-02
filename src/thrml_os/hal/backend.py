"""Abstract backend interface for probabilistic compute units."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, Optional, TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

if TYPE_CHECKING:
    from thrml_os.models.job import SamplingJob
    from thrml_os.models.device import Device
    from thrml_os.models.sample import SampleBatch
    from thrml_os.models.schedule import SamplingSchedule


class BackendType(Enum):
    """Type of compute backend."""
    
    JAX_CPU = "jax_cpu"
    JAX_GPU = "jax_gpu"
    JAX_TPU = "jax_tpu"
    SIMULATOR = "simulator"
    EXTROPIC = "extropic"


@dataclass
class BackendCapabilities:
    """Capabilities of a compute backend."""
    
    backend_type: BackendType
    
    # Supported features
    supports_streaming: bool = True
    supports_batched_sampling: bool = True
    supports_checkpoint: bool = True
    supports_parallel_chains: bool = True
    
    # Limits
    max_graph_nodes: int = 1_000_000
    max_graph_edges: int = 10_000_000
    max_batch_size: int = 1024
    
    # Performance characteristics
    estimated_samples_per_second: Optional[float] = None


@dataclass
class ExecutionContext:
    """Context for a sampling execution."""
    
    key: PRNGKeyArray
    device: Optional[Any] = None  # JAX device or similar
    
    # Callbacks
    on_sample: Optional[Callable[[Array, int], None]] = None
    on_checkpoint: Optional[Callable[[Any, int], None]] = None
    
    # Control
    should_abort: Callable[[], bool] = lambda: False


class Backend(ABC):
    """Abstract base class for compute backends.
    
    A backend is responsible for executing THRML sampling programs
    on specific hardware (CPU, GPU, Extropic, etc.).
    """
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Get the type of this backend."""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Get capabilities of this backend."""
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the backend.
        
        Called once before any sampling operations.
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the backend and release resources."""
        pass
    
    @abstractmethod
    def compile_model(self, model: Any) -> Any:
        """Compile a THRML model for this backend.
        
        Args:
            model: A THRML model (IsingEBM, PGM, etc.)
            
        Returns:
            Compiled representation suitable for this backend
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        compiled_model: Any,
        schedule: "SamplingSchedule",
        initial_state: Optional[Array],
        context: ExecutionContext,
    ) -> Iterator["SampleBatch"]:
        """Execute sampling and yield batches of samples.
        
        This is a generator that yields SampleBatch objects as they
        are produced. Supports streaming and checkpointing.
        
        Args:
            compiled_model: Model compiled with compile_model()
            schedule: Sampling schedule configuration
            initial_state: Optional initial state (for warm start)
            context: Execution context with PRNG key and callbacks
            
        Yields:
            SampleBatch objects containing collected samples
        """
        pass
    
    @abstractmethod
    def checkpoint(self, state: Any) -> bytes:
        """Serialize current state for checkpointing.
        
        Args:
            state: Current sampling state
            
        Returns:
            Serialized state as bytes
        """
        pass
    
    @abstractmethod
    def restore(self, checkpoint: bytes) -> Any:
        """Restore state from checkpoint.
        
        Args:
            checkpoint: Serialized state from checkpoint()
            
        Returns:
            Restored state
        """
        pass
    
    def estimate_memory(self, model: Any, batch_size: int = 1) -> int:
        """Estimate memory requirements in bytes.
        
        Args:
            model: THRML model
            batch_size: Number of parallel chains
            
        Returns:
            Estimated memory in bytes
        """
        # Default implementation - subclasses can override
        return 0
    
    def estimate_time(
        self, 
        model: Any, 
        n_samples: int, 
        n_warmup: int
    ) -> float:
        """Estimate execution time in seconds.
        
        Args:
            model: THRML model
            n_samples: Number of samples
            n_warmup: Warmup steps
            
        Returns:
            Estimated time in seconds
        """
        # Default implementation
        if self.capabilities.estimated_samples_per_second:
            total_steps = n_warmup + n_samples
            return total_steps / self.capabilities.estimated_samples_per_second
        return 0.0
