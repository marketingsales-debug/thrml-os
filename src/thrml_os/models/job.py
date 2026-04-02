"""Job model for THRML-OS sampling tasks."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import jax.numpy as jnp
from jaxtyping import Array


class JobStatus(Enum):
    """Status of a sampling job."""
    
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(Enum):
    """Priority level for job scheduling."""
    
    REALTIME = 0  # Highest priority, preempts others
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4  # Lowest priority, runs when idle


@dataclass
class SamplingJob:
    """A job representing a sampling task to be executed.
    
    Attributes:
        model: The THRML model (EBM, PGM, etc.) to sample from
        n_samples: Number of samples to collect
        n_warmup: Number of warmup/burn-in sweeps
        steps_per_sample: Gibbs sweeps between collected samples
        priority: Job priority level
        deadline: Optional deadline for job completion
        checkpoint_interval: Sweeps between checkpoints (0 = no checkpoints)
        tags: User-defined tags for organization
    """
    
    # Required parameters
    model: Any  # THRML model (EBM, PGM, etc.)
    n_samples: int = 1000
    
    # Sampling configuration
    n_warmup: int = 100
    steps_per_sample: int = 1
    
    # Scheduling
    priority: JobPriority = JobPriority.NORMAL
    deadline: Optional[datetime] = None
    
    # Fault tolerance
    checkpoint_interval: int = 100  # Sweeps between checkpoints
    
    # Metadata
    tags: list[str] = field(default_factory=list)
    
    # Auto-generated fields
    id: str = field(default_factory=lambda: f"job-{uuid.uuid4().hex[:12]}")
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    current_sweep: int = 0
    samples_collected: int = 0
    
    # Results (populated after completion)
    samples: Optional[Array] = None
    energies: Optional[Array] = None
    
    # Error information
    error_message: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate job parameters."""
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if self.n_warmup < 0:
            raise ValueError("n_warmup cannot be negative")
        if self.steps_per_sample <= 0:
            raise ValueError("steps_per_sample must be positive")
        if isinstance(self.priority, str):
            self.priority = JobPriority[self.priority.upper()]
    
    @property
    def total_sweeps(self) -> int:
        """Total number of Gibbs sweeps needed."""
        return self.n_warmup + (self.n_samples * self.steps_per_sample)
    
    @property
    def progress(self) -> float:
        """Job progress as a percentage (0-100)."""
        if self.total_sweeps == 0:
            return 100.0
        return (self.current_sweep / self.total_sweeps) * 100
    
    @property
    def is_terminal(self) -> bool:
        """Whether the job is in a terminal state."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        )
    
    def start(self) -> None:
        """Mark job as started."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.utcnow()
    
    def complete(self, samples: Array, energies: Optional[Array] = None) -> None:
        """Mark job as completed with results."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.samples = samples
        self.energies = energies
        self.samples_collected = samples.shape[0] if samples is not None else 0
    
    def fail(self, error: str) -> None:
        """Mark job as failed."""
        self.status = JobStatus.FAILED
        self.completed_at = datetime.utcnow()
        self.error_message = error
    
    def cancel(self) -> None:
        """Cancel the job."""
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause a running job."""
        if self.status == JobStatus.RUNNING:
            self.status = JobStatus.PAUSED
    
    def resume(self) -> None:
        """Resume a paused job."""
        if self.status == JobStatus.PAUSED:
            self.status = JobStatus.RUNNING
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "status": self.status.value,
            "priority": self.priority.value,
            "n_samples": self.n_samples,
            "n_warmup": self.n_warmup,
            "steps_per_sample": self.steps_per_sample,
            "total_sweeps": self.total_sweeps,
            "current_sweep": self.current_sweep,
            "samples_collected": self.samples_collected,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "error_message": self.error_message,
        }
