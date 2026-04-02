"""Software simulator backend for testing without hardware."""

from __future__ import annotations

import pickle
import time
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


class SimulatorBackend(Backend):
    """Software simulator for testing THRML-OS.
    
    This backend simulates probabilistic sampling without requiring
    actual THRML models. Useful for:
    - Testing the OS layer
    - Prototyping workflows
    - Benchmarking scheduler behavior
    """
    
    def __init__(
        self,
        samples_per_second: float = 1000.0,
        failure_rate: float = 0.0,
        latency_ms: float = 0.0,
    ):
        """Initialize simulator backend.
        
        Args:
            samples_per_second: Simulated sampling throughput
            failure_rate: Probability of random failures (0-1)
            latency_ms: Simulated latency per operation
        """
        self._samples_per_second = samples_per_second
        self._failure_rate = failure_rate
        self._latency_ms = latency_ms
        self._initialized = False
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.SIMULATOR
    
    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_type=BackendType.SIMULATOR,
            supports_streaming=True,
            supports_batched_sampling=True,
            supports_checkpoint=True,
            supports_parallel_chains=True,
            max_graph_nodes=100_000,
            max_graph_edges=1_000_000,
            max_batch_size=1024,
            estimated_samples_per_second=self._samples_per_second,
        )
    
    def initialize(self) -> None:
        """Initialize the simulator."""
        self._initialized = True
    
    def shutdown(self) -> None:
        """Shutdown the simulator."""
        self._initialized = False
    
    def compile_model(self, model: Any) -> dict[str, Any]:
        """Compile model to simulator representation.
        
        For the simulator, we just extract key parameters.
        """
        # Extract model parameters
        compiled = {
            "n_nodes": 100,  # Default
            "n_edges": 0,
            "state_dtype": jnp.float32,
        }
        
        if hasattr(model, 'nodes'):
            compiled["n_nodes"] = len(model.nodes)
        if hasattr(model, 'edges'):
            compiled["n_edges"] = len(model.edges)
        if hasattr(model, 'biases'):
            compiled["biases"] = model.biases
        if hasattr(model, 'weights'):
            compiled["weights"] = model.weights
        if hasattr(model, 'beta'):
            compiled["beta"] = model.beta
            
        return compiled
    
    def sample(
        self,
        compiled_model: dict[str, Any],
        schedule: SamplingSchedule,
        initial_state: Optional[Array],
        context: ExecutionContext,
    ) -> Iterator[SampleBatch]:
        """Simulate sampling from the model.
        
        Generates random samples with simulated timing to
        match configured throughput.
        """
        n_nodes = compiled_model.get("n_nodes", 100)
        beta = compiled_model.get("beta", jnp.array(1.0))
        
        # Initialize state
        key = context.key
        if initial_state is None:
            key, init_key = jax.random.split(key)
            state = jax.random.choice(init_key, jnp.array([-1, 1]), shape=(n_nodes,))
        else:
            state = initial_state
        
        # Simulate warmup
        warmup_time = schedule.n_warmup / self._samples_per_second
        if self._latency_ms > 0:
            warmup_time += self._latency_ms / 1000
        
        # Check for abort during warmup
        if context.should_abort():
            return
        
        # Simulate warmup delay (scaled down for testing)
        time.sleep(min(warmup_time * 0.01, 0.1))
        
        # Generate samples in batches
        batch_size = min(100, schedule.n_samples)
        n_batches = (schedule.n_samples + batch_size - 1) // batch_size
        
        samples_generated = 0
        
        for batch_idx in range(n_batches):
            # Check for abort
            if context.should_abort():
                break
            
            # Calculate batch size
            remaining = schedule.n_samples - samples_generated
            current_batch_size = min(batch_size, remaining)
            
            # Generate random "samples" (simulating Ising spins)
            key, sample_key = jax.random.split(key)
            batch_samples = jax.random.choice(
                sample_key,
                jnp.array([-1, 1]),
                shape=(current_batch_size, n_nodes),
            )
            
            # Simulate energies (random for now)
            key, energy_key = jax.random.split(key)
            energies = jax.random.uniform(
                energy_key,
                shape=(current_batch_size,),
                minval=-n_nodes,
                maxval=0,
            )
            
            # Simulate timing
            batch_time = current_batch_size / self._samples_per_second
            time.sleep(min(batch_time * 0.01, 0.05))  # Scaled down
            
            # Check for random failures
            if self._failure_rate > 0:
                key, fail_key = jax.random.split(key)
                if jax.random.uniform(fail_key) < self._failure_rate:
                    raise RuntimeError("Simulated hardware failure")
            
            # Create batch
            batch = SampleBatch(
                samples=batch_samples,
                energies=energies,
                batch_index=batch_idx,
                collection_time_ms=batch_time * 1000,
            )
            
            samples_generated += current_batch_size
            
            # Call sample callback if provided
            if context.on_sample is not None:
                context.on_sample(batch_samples, samples_generated)
            
            yield batch
    
    def checkpoint(self, state: Any) -> bytes:
        """Serialize state."""
        return pickle.dumps(state)
    
    def restore(self, checkpoint: bytes) -> Any:
        """Restore state."""
        return pickle.loads(checkpoint)
    
    def estimate_memory(self, model: Any, batch_size: int = 1) -> int:
        """Estimate memory (always returns a fixed small amount)."""
        compiled = self.compile_model(model)
        n_nodes = compiled.get("n_nodes", 100)
        return n_nodes * batch_size * 4  # 4 bytes per float32
