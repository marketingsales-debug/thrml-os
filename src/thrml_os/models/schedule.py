"""Sampling schedule configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Callable

import jax.numpy as jnp
from jaxtyping import Array, Float


@dataclass
class SamplingSchedule:
    """Configuration for a sampling run.
    
    Defines the number of samples, warmup period, and sampling interval.
    Can also include temperature/beta schedules for annealing.
    """
    
    n_warmup: int = 100
    n_samples: int = 1000
    steps_per_sample: int = 1
    
    # Temperature scheduling
    initial_beta: float = 1.0
    final_beta: float = 1.0
    beta_schedule: Optional[str] = None  # "linear", "exponential", "cosine"
    
    @property
    def total_steps(self) -> int:
        """Total number of Gibbs sweeps."""
        return self.n_warmup + (self.n_samples * self.steps_per_sample)
    
    def get_beta(self, step: int) -> float:
        """Get beta (inverse temperature) at a given step.
        
        Args:
            step: Current step number
            
        Returns:
            Beta value for this step
        """
        if self.beta_schedule is None or self.initial_beta == self.final_beta:
            return self.final_beta
        
        # During warmup, interpolate beta
        if step < self.n_warmup:
            progress = step / self.n_warmup
        else:
            progress = 1.0
        
        if self.beta_schedule == "linear":
            return self.initial_beta + progress * (self.final_beta - self.initial_beta)
        elif self.beta_schedule == "exponential":
            return self.initial_beta * (self.final_beta / self.initial_beta) ** progress
        elif self.beta_schedule == "cosine":
            # Cosine annealing
            cos_progress = 0.5 * (1 - jnp.cos(jnp.pi * progress))
            return self.initial_beta + cos_progress * (self.final_beta - self.initial_beta)
        else:
            return self.final_beta
    
    def get_beta_array(self) -> Array:
        """Get beta values for all steps as an array.
        
        Returns:
            Array of beta values, one per step
        """
        return jnp.array([self.get_beta(i) for i in range(self.total_steps)])


@dataclass 
class AnnealingSchedule(SamplingSchedule):
    """Schedule with simulated annealing for optimization problems."""
    
    initial_beta: float = 0.1  # Start hot (high temperature)
    final_beta: float = 10.0   # End cold (low temperature)
    beta_schedule: str = "exponential"
    
    # Annealing-specific
    restart_threshold: Optional[float] = None  # Energy threshold to restart
    max_restarts: int = 0


@dataclass
class ParallelTemperingSchedule:
    """Schedule for parallel tempering (replica exchange)."""
    
    n_replicas: int = 4
    beta_min: float = 0.1
    beta_max: float = 10.0
    exchange_interval: int = 10
    
    n_warmup: int = 100
    n_samples: int = 1000
    steps_per_sample: int = 1
    
    @property
    def betas(self) -> Array:
        """Get beta values for all replicas (geometric spacing)."""
        return jnp.geomspace(self.beta_min, self.beta_max, self.n_replicas)
