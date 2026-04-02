"""Sample batch data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Array


@dataclass
class SampleBatch:
    """A batch of samples collected from a sampling job.
    
    Stores samples along with metadata like energies, timestamps,
    and convergence diagnostics.
    """
    
    # Core data
    samples: Array  # Shape: (n_samples, *state_shape)
    
    # Metadata
    job_id: str = ""
    batch_index: int = 0
    
    # Per-sample information
    energies: Optional[Array] = None  # Shape: (n_samples,)
    sweep_indices: Optional[Array] = None  # Which sweep each sample came from
    
    # Timing
    collected_at: datetime = field(default_factory=datetime.utcnow)
    collection_time_ms: float = 0.0
    
    # Diagnostics
    acceptance_rate: Optional[float] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples in batch."""
        return self.samples.shape[0]
    
    @property
    def state_shape(self) -> tuple[int, ...]:
        """Shape of individual sample states."""
        return self.samples.shape[1:]
    
    @property
    def mean_energy(self) -> Optional[float]:
        """Mean energy of samples in batch."""
        if self.energies is None:
            return None
        return float(jnp.mean(self.energies))
    
    @property
    def energy_variance(self) -> Optional[float]:
        """Variance of energies in batch."""
        if self.energies is None:
            return None
        return float(jnp.var(self.energies))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without large arrays)."""
        return {
            "job_id": self.job_id,
            "batch_index": self.batch_index,
            "n_samples": self.n_samples,
            "state_shape": self.state_shape,
            "mean_energy": self.mean_energy,
            "energy_variance": self.energy_variance,
            "acceptance_rate": self.acceptance_rate,
            "collected_at": self.collected_at.isoformat(),
            "collection_time_ms": self.collection_time_ms,
        }


@dataclass
class SampleStream:
    """Iterator over streaming samples from a running job."""
    
    job_id: str
    batches: list[SampleBatch] = field(default_factory=list)
    is_complete: bool = False
    
    def add_batch(self, batch: SampleBatch) -> None:
        """Add a new batch of samples."""
        batch.job_id = self.job_id
        batch.batch_index = len(self.batches)
        self.batches.append(batch)
    
    @property
    def total_samples(self) -> int:
        """Total samples collected so far."""
        return sum(b.n_samples for b in self.batches)
    
    @property
    def all_samples(self) -> Optional[Array]:
        """Concatenate all samples into single array."""
        if not self.batches:
            return None
        return jnp.concatenate([b.samples for b in self.batches], axis=0)
    
    @property
    def all_energies(self) -> Optional[Array]:
        """Concatenate all energies."""
        if not self.batches or self.batches[0].energies is None:
            return None
        return jnp.concatenate([b.energies for b in self.batches], axis=0)
