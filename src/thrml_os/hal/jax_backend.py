"""JAX-based backend for CPU and GPU execution."""

from __future__ import annotations

import pickle
from typing import Any, Iterator, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Array

from thrml_os.hal.backend import (
    Backend,
    BackendType,
    BackendCapabilities,
    ExecutionContext,
)
from thrml_os.models.schedule import SamplingSchedule
from thrml_os.models.sample import SampleBatch


class JAXBackend(Backend):
    """Backend using JAX for CPU/GPU/TPU execution.
    
    This backend uses THRML's native JAX implementation for sampling.
    Supports CPU, GPU (CUDA/ROCm), and TPU devices.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize JAX backend.
        
        Args:
            device: Device specification ("cpu", "gpu", "tpu:0", etc.)
                   If None, uses JAX's default device.
        """
        self._device_spec = device
        self._device: Optional[jax.Device] = None
        self._initialized = False
        self._backend_type: Optional[BackendType] = None
    
    @property
    def backend_type(self) -> BackendType:
        """Get the type of this backend."""
        if self._backend_type is None:
            # Determine from device
            if self._device is None:
                return BackendType.JAX_CPU
            platform = self._device.platform
            if platform == "gpu":
                return BackendType.JAX_GPU
            elif platform == "tpu":
                return BackendType.JAX_TPU
            else:
                return BackendType.JAX_CPU
        return self._backend_type
    
    @property
    def capabilities(self) -> BackendCapabilities:
        """Get capabilities of this backend."""
        is_gpu = self.backend_type == BackendType.JAX_GPU
        is_tpu = self.backend_type == BackendType.JAX_TPU
        
        return BackendCapabilities(
            backend_type=self.backend_type,
            supports_streaming=True,
            supports_batched_sampling=True,
            supports_checkpoint=True,
            supports_parallel_chains=True,
            max_graph_nodes=10_000_000 if (is_gpu or is_tpu) else 1_000_000,
            max_graph_edges=100_000_000 if (is_gpu or is_tpu) else 10_000_000,
            max_batch_size=4096 if is_gpu else (8192 if is_tpu else 256),
            estimated_samples_per_second=100000 if is_gpu else (500000 if is_tpu else 10000),
        )
    
    def initialize(self) -> None:
        """Initialize the JAX backend."""
        if self._initialized:
            return
        
        # Select device
        if self._device_spec is None:
            self._device = jax.devices()[0]
        elif self._device_spec == "cpu":
            self._device = jax.devices("cpu")[0]
        elif self._device_spec.startswith("gpu"):
            gpu_devices = jax.devices("gpu")
            if ":" in self._device_spec:
                idx = int(self._device_spec.split(":")[1])
                self._device = gpu_devices[idx]
            else:
                self._device = gpu_devices[0]
        elif self._device_spec.startswith("tpu"):
            tpu_devices = jax.devices("tpu")
            if ":" in self._device_spec:
                idx = int(self._device_spec.split(":")[1])
                self._device = tpu_devices[idx]
            else:
                self._device = tpu_devices[0]
        else:
            self._device = jax.devices()[0]
        
        # Update backend type based on actual device
        platform = self._device.platform
        if platform == "gpu":
            self._backend_type = BackendType.JAX_GPU
        elif platform == "tpu":
            self._backend_type = BackendType.JAX_TPU
        else:
            self._backend_type = BackendType.JAX_CPU
        
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the JAX backend."""
        self._initialized = False
        self._device = None
    
    def compile_model(self, model: Any) -> Any:
        """Compile a THRML model for JAX execution.
        
        Args:
            model: A THRML model (IsingEBM, DiscreteEBM, etc.)
            
        Returns:
            The model itself (THRML models are already JAX-compatible)
        """
        # THRML models are already JAX-native
        # We could add JIT compilation here if needed
        return model
    
    def sample(
        self,
        compiled_model: Any,
        schedule: SamplingSchedule,
        initial_state: Optional[Array],
        context: ExecutionContext,
    ) -> Iterator[SampleBatch]:
        """Execute sampling using THRML's sample_states function.
        
        Args:
            compiled_model: THRML model
            schedule: Sampling schedule
            initial_state: Optional initial state
            context: Execution context
            
        Yields:
            SampleBatch objects
        """
        import time
        
        # Import THRML components
        try:
            from thrml import Block, sample_states
            from thrml import SamplingSchedule as THRMLSchedule
        except ImportError:
            raise ImportError(
                "THRML is required for JAX backend. "
                "Install with: pip install thrml"
            )
        
        # Get model components
        model = compiled_model
        
        # Determine blocks for sampling
        # For EBMs, we need to create blocks from nodes
        if hasattr(model, 'nodes'):
            nodes = model.nodes
            # Two-color block Gibbs (checkerboard pattern)
            free_blocks = [
                Block(nodes[::2]),  # Even nodes
                Block(nodes[1::2]),  # Odd nodes
            ]
        else:
            raise ValueError("Model must have 'nodes' attribute")
        
        # Create THRML schedule
        thrml_schedule = THRMLSchedule(
            n_warmup=schedule.n_warmup,
            n_samples=schedule.n_samples,
            steps_per_sample=schedule.steps_per_sample,
        )
        
        # Initialize state if not provided
        if initial_state is None:
            from thrml.models import hinton_init
            key, init_key = jax.random.split(context.key)
            initial_state = hinton_init(init_key, model, free_blocks, ())
        
        # Build sampling program
        if hasattr(model, '__class__') and 'Ising' in model.__class__.__name__:
            from thrml.models import IsingSamplingProgram
            program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        else:
            # Generic EBM program
            from thrml.models import EBMSamplingProgram
            program = EBMSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        # Run sampling
        start_time = time.time()
        
        # Use THRML's sample_states
        key, sample_key = jax.random.split(context.key)
        
        raw_samples = sample_states(
            sample_key,
            program,
            thrml_schedule,
            initial_state,
            [],  # No observed blocks
            [Block(nodes)],  # Collect all nodes
        )

        # sample_states returns a list of arrays, one per observe block
        samples = raw_samples[0]

        elapsed = time.time() - start_time

        # Yield as a single batch
        batch = SampleBatch(
            samples=samples,
            energies=None,
            collection_time_ms=elapsed * 1000,
        )
        
        yield batch
    
    def checkpoint(self, state: Any) -> bytes:
        """Serialize state for checkpointing."""
        # Convert JAX arrays to numpy for pickling
        def to_numpy(x: Any) -> Any:
            if isinstance(x, jnp.ndarray):
                return jnp.asarray(x)
            return x
        
        state_np = jax.tree.map(to_numpy, state)
        return pickle.dumps(state_np)
    
    def restore(self, checkpoint: bytes) -> Any:
        """Restore state from checkpoint."""
        state_np = pickle.loads(checkpoint)
        
        # Convert back to JAX arrays on correct device
        def to_jax(x: Any) -> Any:
            if hasattr(x, 'shape'):  # numpy-like array
                arr = jnp.array(x)
                if self._device is not None:
                    arr = jax.device_put(arr, self._device)
                return arr
            return x
        
        return jax.tree.map(to_jax, state_np)
    
    def estimate_memory(self, model: Any, batch_size: int = 1) -> int:
        """Estimate memory requirements."""
        if not hasattr(model, 'nodes'):
            return 0
        
        n_nodes = len(model.nodes)
        n_edges = len(model.edges) if hasattr(model, 'edges') else 0
        
        # Rough estimate: 4 bytes per float32
        # State: n_nodes * batch_size
        # Graph: n_edges * 2 (indices) + n_edges (weights)
        state_bytes = n_nodes * batch_size * 4
        graph_bytes = n_edges * 3 * 4
        
        # Add buffer for intermediates
        return int((state_bytes + graph_bytes) * 1.5)
