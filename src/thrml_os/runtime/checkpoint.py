"""Checkpoint management for fault tolerance."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp


@dataclass
class Checkpoint:
    """A checkpoint of job state."""
    
    job_id: str
    sweep_index: int
    state: Any
    timestamp: float
    checksum: str
    
    def verify(self) -> bool:
        """Verify checkpoint integrity."""
        computed = hashlib.sha256(pickle.dumps(self.state)).hexdigest()[:16]
        return computed == self.checksum


class CheckpointManager:
    """Manages checkpoints for fault-tolerant execution.
    
    Features:
    - Incremental checkpointing
    - Compression
    - Integrity verification
    - Automatic cleanup
    """
    
    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        max_checkpoints: int = 5,
        compression: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage
            max_checkpoints: Maximum checkpoints to keep per job
            compression: Enable compression
        """
        self._dir = checkpoint_dir or Path.home() / ".thrml-os" / "checkpoints"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_checkpoints = max_checkpoints
        self._compression = compression
    
    def save(
        self,
        job_id: str,
        sweep_index: int,
        state: Any,
    ) -> Checkpoint:
        """Save a checkpoint.
        
        Args:
            job_id: Job identifier
            sweep_index: Current sweep number
            state: State to checkpoint
            
        Returns:
            Checkpoint object
        """
        # Compute checksum
        state_bytes = pickle.dumps(state)
        checksum = hashlib.sha256(state_bytes).hexdigest()[:16]
        
        # Create checkpoint
        checkpoint = Checkpoint(
            job_id=job_id,
            sweep_index=sweep_index,
            state=state,
            timestamp=time.time(),
            checksum=checksum,
        )
        
        # Save to disk
        job_dir = self._dir / job_id
        job_dir.mkdir(exist_ok=True)
        
        filename = f"checkpoint_{sweep_index:08d}.pkl"
        filepath = job_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Cleanup old checkpoints
        self._cleanup(job_id)
        
        return checkpoint
    
    def load(
        self,
        job_id: str,
        sweep_index: Optional[int] = None,
    ) -> Optional[Checkpoint]:
        """Load a checkpoint.
        
        Args:
            job_id: Job identifier
            sweep_index: Specific sweep (None for latest)
            
        Returns:
            Checkpoint or None if not found
        """
        job_dir = self._dir / job_id
        if not job_dir.exists():
            return None
        
        # Find checkpoint files
        checkpoints = sorted(job_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return None
        
        if sweep_index is not None:
            # Find specific checkpoint
            filename = f"checkpoint_{sweep_index:08d}.pkl"
            filepath = job_dir / filename
            if not filepath.exists():
                return None
        else:
            # Use latest
            filepath = checkpoints[-1]
        
        # Load checkpoint
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Verify integrity
        if not checkpoint.verify():
            raise ValueError(f"Checkpoint corrupted: {filepath}")
        
        return checkpoint
    
    def list_checkpoints(self, job_id: str) -> list[int]:
        """List available checkpoint sweep indices.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of sweep indices
        """
        job_dir = self._dir / job_id
        if not job_dir.exists():
            return []
        
        indices = []
        for path in job_dir.glob("checkpoint_*.pkl"):
            # Extract sweep index from filename
            name = path.stem
            index = int(name.split("_")[1])
            indices.append(index)
        
        return sorted(indices)
    
    def delete(self, job_id: str) -> None:
        """Delete all checkpoints for a job.
        
        Args:
            job_id: Job identifier
        """
        import shutil
        
        job_dir = self._dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
    
    def _cleanup(self, job_id: str) -> None:
        """Remove old checkpoints beyond the limit."""
        job_dir = self._dir / job_id
        if not job_dir.exists():
            return
        
        checkpoints = sorted(job_dir.glob("checkpoint_*.pkl"))
        
        # Remove oldest if over limit
        while len(checkpoints) > self._max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
