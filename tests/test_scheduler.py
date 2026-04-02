"""Tests for the job scheduler."""

import pytest
import threading
import time

import jax
import jax.numpy as jnp

from thrml_os.scheduler.queue import JobQueue, PriorityJobQueue
from thrml_os.scheduler.scheduler import Scheduler
from thrml_os.models.job import SamplingJob, JobStatus, JobPriority
from thrml_os.hal.simulator import SimulatorBackend


class TestJobQueue:
    """Tests for FIFO JobQueue."""
    
    def test_push_pop(self):
        """Test basic push/pop operations."""
        queue = JobQueue()
        
        job1 = SamplingJob(model=None, n_samples=10)
        job2 = SamplingJob(model=None, n_samples=20)
        
        queue.push(job1)
        queue.push(job2)
        
        assert len(queue) == 2
        
        result = queue.pop()
        assert result.id == job1.id
        
        result = queue.pop()
        assert result.id == job2.id
        
        assert len(queue) == 0
        assert queue.pop() is None
    
    def test_peek(self):
        """Test peek without removal."""
        queue = JobQueue()
        job = SamplingJob(model=None)
        
        queue.push(job)
        
        peeked = queue.peek()
        assert peeked.id == job.id
        assert len(queue) == 1  # Still in queue
    
    def test_remove(self):
        """Test removal by ID."""
        queue = JobQueue()
        
        job1 = SamplingJob(model=None)
        job2 = SamplingJob(model=None)
        
        queue.push(job1)
        queue.push(job2)
        
        assert queue.remove(job1.id)
        assert len(queue) == 1
        
        assert not queue.remove("nonexistent")


class TestPriorityJobQueue:
    """Tests for PriorityJobQueue."""
    
    def test_priority_ordering(self):
        """Test jobs come out in priority order."""
        queue = PriorityJobQueue()
        
        low = SamplingJob(model=None, priority=JobPriority.LOW)
        high = SamplingJob(model=None, priority=JobPriority.HIGH)
        normal = SamplingJob(model=None, priority=JobPriority.NORMAL)
        
        # Add in random order
        queue.push(low)
        queue.push(high)
        queue.push(normal)
        
        # Should come out: HIGH, NORMAL, LOW
        assert queue.pop().priority == JobPriority.HIGH
        assert queue.pop().priority == JobPriority.NORMAL
        assert queue.pop().priority == JobPriority.LOW
    
    def test_same_priority_fifo(self):
        """Test FIFO within same priority."""
        queue = PriorityJobQueue()
        
        job1 = SamplingJob(model=None, priority=JobPriority.NORMAL)
        job2 = SamplingJob(model=None, priority=JobPriority.NORMAL)
        job3 = SamplingJob(model=None, priority=JobPriority.NORMAL)
        
        queue.push(job1)
        queue.push(job2)
        queue.push(job3)
        
        assert queue.pop().id == job1.id
        assert queue.pop().id == job2.id
        assert queue.pop().id == job3.id
    
    def test_update_priority(self):
        """Test priority update."""
        queue = PriorityJobQueue()
        
        low = SamplingJob(model=None, priority=JobPriority.LOW)
        queue.push(low)
        
        # Upgrade to HIGH
        queue.update_priority(low.id, JobPriority.HIGH)
        
        result = queue.pop()
        assert result.priority == JobPriority.HIGH


class TestScheduler:
    """Tests for the main Scheduler."""
    
    def test_start_stop(self):
        """Test scheduler lifecycle."""
        backend = SimulatorBackend()
        scheduler = Scheduler(backend=backend)
        
        scheduler.start()
        assert scheduler._running_flag.is_set()
        
        scheduler.stop()
        assert not scheduler._running_flag.is_set()
    
    def test_submit_job(self):
        """Test job submission."""
        backend = SimulatorBackend(samples_per_second=10000)
        scheduler = Scheduler(backend=backend)
        scheduler.start()
        
        try:
            class MockModel:
                nodes = list(range(5))
                edges = []
            
            job = SamplingJob(model=MockModel(), n_samples=10, n_warmup=5)
            job_id = scheduler.submit(job)
            
            assert job_id is not None
            assert job_id.startswith("job-")
            
            # Wait for completion
            time.sleep(0.5)
            
            result = scheduler.get_job(job_id)
            assert result is not None
        finally:
            scheduler.stop()
    
    def test_list_devices(self):
        """Test device listing."""
        scheduler = Scheduler()
        scheduler.start()
        
        try:
            devices = scheduler.list_devices()
            assert len(devices) >= 1  # At least CPU
        finally:
            scheduler.stop()
    
    def test_cancel_queued_job(self):
        """Test cancelling a queued job."""
        backend = SimulatorBackend(samples_per_second=1)  # Very slow
        scheduler = Scheduler(backend=backend, max_concurrent_jobs=1)
        scheduler.start()
        
        try:
            class MockModel:
                nodes = list(range(5))
                edges = []
            
            # Submit first job (will start running)
            job1 = SamplingJob(model=MockModel(), n_samples=1000)
            scheduler.submit(job1)
            
            # Submit second job (will be queued)
            job2 = SamplingJob(model=MockModel(), n_samples=100)
            job2_id = scheduler.submit(job2)
            
            # Cancel the queued job
            result = scheduler.cancel(job2_id)
            assert result
        finally:
            scheduler.stop(wait=False)
    
    def test_job_completion_callback(self):
        """Test completion callback."""
        backend = SimulatorBackend(samples_per_second=100000)
        scheduler = Scheduler(backend=backend)
        
        completed_jobs = []
        scheduler.on_job_complete(lambda job: completed_jobs.append(job))
        
        scheduler.start()
        
        try:
            class MockModel:
                nodes = list(range(3))
                edges = []
            
            job = SamplingJob(model=MockModel(), n_samples=10, n_warmup=0)
            scheduler.submit(job)
            
            # Wait for completion
            time.sleep(0.5)
            
            assert len(completed_jobs) == 1
            assert completed_jobs[0].status == JobStatus.COMPLETED
        finally:
            scheduler.stop()
