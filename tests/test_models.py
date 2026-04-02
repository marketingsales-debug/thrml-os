"""Tests for core data models."""

import pytest
from datetime import datetime

from thrml_os.models.job import SamplingJob, JobStatus, JobPriority
from thrml_os.models.device import Device, DeviceType, DeviceStatus, DeviceCapabilities
from thrml_os.models.schedule import SamplingSchedule, AnnealingSchedule


class TestSamplingJob:
    """Tests for SamplingJob."""
    
    def test_create_job(self):
        """Test basic job creation."""
        job = SamplingJob(
            model=None,  # Mock model
            n_samples=500,
            n_warmup=50,
        )
        
        assert job.n_samples == 500
        assert job.n_warmup == 50
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.id.startswith("job-")
    
    def test_job_total_sweeps(self):
        """Test total sweeps calculation."""
        job = SamplingJob(
            model=None,
            n_samples=100,
            n_warmup=20,
            steps_per_sample=2,
        )
        
        # 20 warmup + (100 samples * 2 steps) = 220
        assert job.total_sweeps == 220
    
    def test_job_progress(self):
        """Test progress calculation."""
        job = SamplingJob(model=None, n_samples=100, n_warmup=0)
        
        assert job.progress == 0.0
        
        job.current_sweep = 50
        assert job.progress == 50.0
        
        job.current_sweep = 100
        assert job.progress == 100.0
    
    def test_job_lifecycle(self):
        """Test job state transitions."""
        job = SamplingJob(model=None)
        
        assert job.status == JobStatus.PENDING
        assert job.started_at is None
        
        job.start()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None
        
        import jax.numpy as jnp
        samples = jnp.zeros((10, 5))
        job.complete(samples)
        
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.samples is not None
        assert job.is_terminal
    
    def test_job_failure(self):
        """Test job failure handling."""
        job = SamplingJob(model=None)
        job.start()
        job.fail("Test error")
        
        assert job.status == JobStatus.FAILED
        assert job.error_message == "Test error"
        assert job.is_terminal
    
    def test_job_priority_from_string(self):
        """Test priority can be set from string."""
        job = SamplingJob(model=None, priority="high")
        assert job.priority == JobPriority.HIGH
    
    def test_invalid_params(self):
        """Test validation of invalid parameters."""
        with pytest.raises(ValueError):
            SamplingJob(model=None, n_samples=0)
        
        with pytest.raises(ValueError):
            SamplingJob(model=None, n_warmup=-1)


class TestDevice:
    """Tests for Device."""
    
    def test_create_device(self):
        """Test device creation."""
        device = Device(
            name="Test GPU",
            device_type=DeviceType.GPU_CUDA,
        )
        
        assert device.name == "Test GPU"
        assert device.device_type == DeviceType.GPU_CUDA
        assert device.status == DeviceStatus.AVAILABLE
        assert device.id.startswith("pcu-")
    
    def test_device_assignment(self):
        """Test job assignment to device."""
        device = Device()
        
        assert device.is_available
        assert not device.is_busy
        
        device.assign_job("job-123")
        
        assert not device.is_available
        assert device.is_busy
        assert device.current_job_id == "job-123"
        
        device.release_job()
        
        assert device.is_available
        assert device.current_job_id is None
        assert device.jobs_completed == 1
    
    def test_device_utilization(self):
        """Test memory utilization calculation."""
        device = Device(
            capabilities=DeviceCapabilities(
                total_memory_gb=16.0,
                available_memory_gb=12.0,
            )
        )
        
        # (16 - 12) / 16 * 100 = 25%
        assert device.utilization == 25.0


class TestSamplingSchedule:
    """Tests for SamplingSchedule."""
    
    def test_basic_schedule(self):
        """Test basic schedule creation."""
        schedule = SamplingSchedule(
            n_warmup=100,
            n_samples=1000,
            steps_per_sample=2,
        )
        
        assert schedule.total_steps == 100 + 1000 * 2
    
    def test_constant_beta(self):
        """Test constant temperature."""
        schedule = SamplingSchedule(initial_beta=1.0, final_beta=1.0)
        
        assert schedule.get_beta(0) == 1.0
        assert schedule.get_beta(50) == 1.0
        assert schedule.get_beta(100) == 1.0
    
    def test_linear_annealing(self):
        """Test linear beta schedule."""
        schedule = SamplingSchedule(
            n_warmup=100,
            initial_beta=0.1,
            final_beta=1.0,
            beta_schedule="linear",
        )
        
        assert schedule.get_beta(0) == pytest.approx(0.1)
        assert schedule.get_beta(50) == pytest.approx(0.55)
        assert schedule.get_beta(100) == pytest.approx(1.0)
    
    def test_annealing_schedule(self):
        """Test AnnealingSchedule preset."""
        schedule = AnnealingSchedule()
        
        assert schedule.initial_beta == 0.1
        assert schedule.final_beta == 10.0
        assert schedule.beta_schedule == "exponential"
