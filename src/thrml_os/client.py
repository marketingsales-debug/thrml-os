"""THRML-OS Python client for programmatic access."""

from __future__ import annotations

from typing import Any, Iterator, List, Optional

import jax

from thrml_os.hal.backend import Backend, ExecutionContext
from thrml_os.hal.registry import get_default_backend
from thrml_os.models.job import SamplingJob, JobStatus, JobPriority
from thrml_os.models.device import Device, detect_devices
from thrml_os.models.schedule import SamplingSchedule
from thrml_os.models.sample import SampleBatch, SampleStream
from thrml_os.scheduler import Scheduler


class THRMLClient:
    """High-level client for THRML-OS.
    
    Provides a simple interface for submitting and managing
    probabilistic sampling jobs.
    
    Example:
        ```python
        from thrml_os import THRMLClient, SamplingJob
        from thrml.models import IsingEBM
        
        # Create model
        model = IsingEBM(nodes, edges, biases, weights, beta)
        
        # Create client and submit job
        client = THRMLClient()
        job = SamplingJob(model=model, n_samples=1000)
        result = client.run(job)
        
        print(f"Collected {result.samples.shape[0]} samples")
        ```
    """
    
    def __init__(
        self,
        backend: Optional[Backend] = None,
        device: Optional[str] = None,
    ):
        """Initialize the client.
        
        Args:
            backend: Compute backend (auto-detects if None)
            device: Device specification (e.g., "cpu", "gpu:0")
        """
        self._backend = backend
        self._device = device
        self._scheduler: Optional[Scheduler] = None
    
    def _get_scheduler(self) -> Scheduler:
        """Get or create the scheduler."""
        if self._scheduler is None:
            self._scheduler = Scheduler(backend=self._backend)
            self._scheduler.start()
        return self._scheduler
    
    def submit(self, job: SamplingJob) -> str:
        """Submit a job for asynchronous execution.
        
        Args:
            job: Sampling job to execute
            
        Returns:
            Job ID for tracking
        """
        scheduler = self._get_scheduler()
        return scheduler.submit(job)
    
    def run(
        self,
        job: SamplingJob,
        progress_callback: Optional[callable] = None,
    ) -> SamplingJob:
        """Run a job synchronously and wait for completion.
        
        Args:
            job: Sampling job to execute
            progress_callback: Optional callback(job) for progress updates
            
        Returns:
            Completed job with samples
        """
        import time
        
        scheduler = self._get_scheduler()
        job_id = scheduler.submit(job)
        
        # Poll for completion
        while True:
            current = scheduler.get_job(job_id)
            if current is None:
                raise RuntimeError(f"Job {job_id} not found")
            
            if progress_callback:
                progress_callback(current)
            
            if current.is_terminal:
                return current
            
            time.sleep(0.01)  # Small sleep to avoid busy waiting
    
    def run_simple(
        self,
        model: Any,
        n_samples: int = 1000,
        n_warmup: int = 100,
        **kwargs,
    ) -> SamplingJob:
        """Simplified interface for running a sampling job.
        
        Args:
            model: THRML model to sample from
            n_samples: Number of samples to collect
            n_warmup: Warmup sweeps
            **kwargs: Additional job parameters
            
        Returns:
            Completed job with samples
        """
        job = SamplingJob(
            model=model,
            n_samples=n_samples,
            n_warmup=n_warmup,
            **kwargs,
        )
        return self.run(job)
    
    def stream(self, job: SamplingJob) -> Iterator[SampleBatch]:
        """Stream samples from a job as they are generated.
        
        Args:
            job: Sampling job to execute
            
        Yields:
            SampleBatch objects as they are collected
        """
        import time
        
        scheduler = self._get_scheduler()
        job_id = scheduler.submit(job)
        
        last_batch = 0
        
        while True:
            stream = scheduler.get_samples(job_id)
            if stream is None:
                break
            
            # Yield any new batches
            while last_batch < len(stream.batches):
                yield stream.batches[last_batch]
                last_batch += 1
            
            # Check if complete
            if stream.is_complete:
                break
            
            time.sleep(0.01)
    
    def get_job(self, job_id: str) -> Optional[SamplingJob]:
        """Get a job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job or None if not found
        """
        scheduler = self._get_scheduler()
        return scheduler.get_job(job_id)
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was cancelled
        """
        scheduler = self._get_scheduler()
        return scheduler.cancel(job_id)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[SamplingJob]:
        """List jobs.
        
        Args:
            status: Filter by status
            limit: Maximum jobs to return
            
        Returns:
            List of jobs
        """
        scheduler = self._get_scheduler()
        return scheduler.list_jobs(status=status, limit=limit)
    
    def list_devices(self) -> List[Device]:
        """List available compute devices."""
        return detect_devices()
    
    def close(self) -> None:
        """Close the client and release resources."""
        if self._scheduler:
            self._scheduler.stop()
            self._scheduler = None
    
    def __enter__(self) -> "THRMLClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


# Convenience function for quick sampling
def sample(
    model: Any,
    n_samples: int = 1000,
    n_warmup: int = 100,
    **kwargs,
):
    """Quick sampling function.
    
    Args:
        model: THRML model
        n_samples: Number of samples
        n_warmup: Warmup sweeps
        **kwargs: Additional parameters
        
    Returns:
        Array of samples
    """
    with THRMLClient() as client:
        job = client.run_simple(model, n_samples, n_warmup, **kwargs)
        return job.samples
