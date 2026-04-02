"""Main scheduler for THRML-OS job execution."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import jax

from thrml_os.hal.backend import Backend, ExecutionContext
from thrml_os.hal.registry import get_default_backend
from thrml_os.models.job import SamplingJob, JobStatus, JobPriority
from thrml_os.models.device import Device, detect_devices
from thrml_os.models.schedule import SamplingSchedule
from thrml_os.models.sample import SampleBatch, SampleStream
from thrml_os.scheduler.queue import PriorityJobQueue

logger = logging.getLogger(__name__)


class Scheduler:
    """THRML-OS Job Scheduler.
    
    Manages job queuing, execution, and resource allocation for
    probabilistic sampling workloads.
    
    Features:
    - Priority-based scheduling
    - Multi-device support
    - Job preemption at sweep boundaries
    - Checkpoint/resume for fault tolerance
    - Streaming sample delivery
    """
    
    def __init__(
        self,
        backend: Optional[Backend] = None,
        max_concurrent_jobs: int = 1,
        enable_preemption: bool = True,
    ):
        """Initialize the scheduler.
        
        Args:
            backend: Compute backend to use (auto-detects if None)
            max_concurrent_jobs: Maximum jobs to run concurrently
            enable_preemption: Allow high-priority jobs to preempt lower ones
        """
        self._backend = backend
        self._max_concurrent = max_concurrent_jobs
        self._enable_preemption = enable_preemption
        
        # Job management
        self._queue = PriorityJobQueue()
        self._running: Dict[str, SamplingJob] = {}
        self._completed: Dict[str, SamplingJob] = {}
        self._streams: Dict[str, SampleStream] = {}
        
        # Threading
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._abort_flags: Dict[str, threading.Event] = {}
        
        # State
        self._running_flag = threading.Event()
        self._devices: List[Device] = []
        
        # Callbacks
        self._on_job_complete: Optional[Callable[[SamplingJob], None]] = None
        self._on_sample: Optional[Callable[[str, SampleBatch], None]] = None
    
    def start(self) -> None:
        """Start the scheduler."""
        if self._running_flag.is_set():
            return
        
        # Initialize backend
        if self._backend is None:
            self._backend = get_default_backend()
        
        # Detect devices
        self._devices = detect_devices()
        logger.info(f"Detected {len(self._devices)} compute devices")
        
        # Start executor
        self._executor = ThreadPoolExecutor(max_workers=self._max_concurrent)
        self._running_flag.set()
        
        logger.info("Scheduler started")
    
    def stop(self, wait: bool = True) -> None:
        """Stop the scheduler.
        
        Args:
            wait: If True, wait for running jobs to complete
        """
        self._running_flag.clear()
        
        # Cancel pending jobs
        for job in self._queue.list_jobs():
            job.cancel()
        
        # Signal abort to running jobs
        for event in self._abort_flags.values():
            event.set()
        
        # Shutdown executor
        if self._executor:
            self._executor.shutdown(wait=wait)
            self._executor = None
        
        # Shutdown backend
        if self._backend:
            self._backend.shutdown()
        
        logger.info("Scheduler stopped")
    
    def submit(self, job: SamplingJob) -> str:
        """Submit a job for execution.
        
        Args:
            job: The sampling job to execute
            
        Returns:
            Job ID
        """
        with self._lock:
            # Initialize sample stream
            self._streams[job.id] = SampleStream(job_id=job.id)
            
            # Create abort flag
            self._abort_flags[job.id] = threading.Event()
            
            # Add to queue
            job.status = JobStatus.QUEUED
            self._queue.push(job)
            
            logger.info(f"Job {job.id} submitted (priority: {job.priority.name})")
        
        # Try to schedule immediately
        self._schedule_next()
        
        return job.id
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a job.
        
        Args:
            job_id: ID of job to cancel
            
        Returns:
            True if job was cancelled
        """
        with self._lock:
            # Check if in queue
            if self._queue.remove(job_id):
                logger.info(f"Job {job_id} cancelled (was queued)")
                return True
            
            # Check if running
            if job_id in self._running:
                job = self._running[job_id]
                self._abort_flags[job_id].set()
                job.cancel()
                logger.info(f"Job {job_id} cancelled (was running)")
                return True
            
            return False
    
    def get_job(self, job_id: str) -> Optional[SamplingJob]:
        """Get a job by ID.
        
        Args:
            job_id: Job ID to look up
            
        Returns:
            Job or None if not found
        """
        with self._lock:
            # Check running
            if job_id in self._running:
                return self._running[job_id]
            
            # Check completed
            if job_id in self._completed:
                return self._completed[job_id]
            
            # Check queue
            for job in self._queue.list_jobs():
                if job.id == job_id:
                    return job
            
            return None
    
    def get_samples(self, job_id: str) -> Optional[SampleStream]:
        """Get the sample stream for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            SampleStream or None
        """
        return self._streams.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100,
    ) -> List[SamplingJob]:
        """List jobs, optionally filtered by status.
        
        Args:
            status: Filter by status (None for all)
            limit: Maximum jobs to return
            
        Returns:
            List of jobs
        """
        jobs = []
        
        with self._lock:
            # Running jobs
            jobs.extend(self._running.values())
            
            # Queued jobs
            jobs.extend(self._queue.list_jobs())
            
            # Completed jobs
            jobs.extend(self._completed.values())
        
        # Filter by status
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by created time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    def list_devices(self) -> List[Device]:
        """List available compute devices."""
        return self._devices.copy()
    
    def _schedule_next(self) -> None:
        """Try to schedule the next job from the queue."""
        if not self._running_flag.is_set():
            return
        
        with self._lock:
            # Check if we can run more jobs
            if len(self._running) >= self._max_concurrent:
                return
            
            # Get next job
            job = self._queue.pop()
            if job is None:
                return
            
            # Start execution
            self._running[job.id] = job
            job.start()
            
            # Submit to executor
            future = self._executor.submit(self._execute_job, job)
            self._futures[job.id] = future
            future.add_done_callback(
                lambda f, jid=job.id: self._on_job_done(jid, f)
            )
            
            logger.info(f"Job {job.id} started")
    
    def _execute_job(self, job: SamplingJob) -> None:
        """Execute a sampling job.
        
        This runs in a worker thread.
        """
        try:
            # Compile model
            compiled = self._backend.compile_model(job.model)
            
            # Create schedule
            schedule = SamplingSchedule(
                n_warmup=job.n_warmup,
                n_samples=job.n_samples,
                steps_per_sample=job.steps_per_sample,
            )
            
            # Create execution context
            key = jax.random.key(int(time.time() * 1000) % (2**31))
            abort_flag = self._abort_flags.get(job.id, threading.Event())
            
            context = ExecutionContext(
                key=key,
                should_abort=lambda: abort_flag.is_set(),
                on_sample=lambda samples, count: self._handle_samples(
                    job.id, samples, count
                ),
            )
            
            # Execute sampling
            all_samples = []
            all_energies = []
            
            for batch in self._backend.sample(compiled, schedule, None, context):
                # Add to stream
                stream = self._streams.get(job.id)
                if stream:
                    stream.add_batch(batch)
                
                # Collect results
                all_samples.append(batch.samples)
                if batch.energies is not None:
                    all_energies.append(batch.energies)
                
                # Update progress
                job.samples_collected += batch.n_samples
                
                # Notify callback
                if self._on_sample:
                    self._on_sample(job.id, batch)
                
                # Check abort
                if abort_flag.is_set():
                    break
            
            # Combine results
            import jax.numpy as jnp
            if all_samples:
                samples = jnp.concatenate(all_samples, axis=0)
                energies = jnp.concatenate(all_energies, axis=0) if all_energies else None
                job.complete(samples, energies)
            else:
                job.fail("No samples collected")
            
        except Exception as e:
            logger.exception(f"Job {job.id} failed")
            job.fail(str(e))
    
    def _handle_samples(self, job_id: str, samples: Any, count: int) -> None:
        """Handle new samples from a running job."""
        job = self._running.get(job_id)
        if job:
            job.current_sweep = count
    
    def _on_job_done(self, job_id: str, future: Future) -> None:
        """Callback when a job completes."""
        with self._lock:
            job = self._running.pop(job_id, None)
            if job:
                self._completed[job_id] = job
                
                # Mark stream complete
                stream = self._streams.get(job_id)
                if stream:
                    stream.is_complete = True
                
                # Cleanup
                self._futures.pop(job_id, None)
                self._abort_flags.pop(job_id, None)
                
                logger.info(
                    f"Job {job_id} completed: {job.status.value} "
                    f"({job.samples_collected} samples)"
                )
                
                # Notify callback
                if self._on_job_complete:
                    self._on_job_complete(job)
        
        # Schedule next job
        self._schedule_next()
    
    # Callback setters
    def on_job_complete(self, callback: Callable[[SamplingJob], None]) -> None:
        """Set callback for job completion."""
        self._on_job_complete = callback
    
    def on_sample(self, callback: Callable[[str, SampleBatch], None]) -> None:
        """Set callback for new samples."""
        self._on_sample = callback
