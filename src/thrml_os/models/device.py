"""Device model representing a Probabilistic Compute Unit (PCU)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class DeviceType(Enum):
    """Type of compute device."""
    
    CPU = "cpu"
    GPU_CUDA = "gpu_cuda"
    GPU_ROCM = "gpu_rocm"
    GPU_METAL = "gpu_metal"
    TPU = "tpu"
    EXTROPIC = "extropic"  # Future Extropic hardware
    SIMULATOR = "simulator"


class DeviceStatus(Enum):
    """Operational status of a device."""
    
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    DRAINING = "draining"  # Not accepting new jobs, finishing current


@dataclass
class DeviceCapabilities:
    """Hardware capabilities of a device."""
    
    # Memory
    total_memory_gb: float = 0.0
    available_memory_gb: float = 0.0
    
    # Compute
    compute_units: int = 1  # Cores, SMs, etc.
    clock_speed_mhz: Optional[int] = None
    
    # Specific to probabilistic hardware
    max_nodes: int = 10000  # Max PGM nodes supported
    max_edges: int = 100000  # Max edges supported
    supported_node_types: list[str] = field(default_factory=lambda: ["spin", "discrete"])
    
    # Performance characteristics
    samples_per_second: Optional[float] = None  # Estimated throughput
    energy_per_sample_mj: Optional[float] = None  # Energy efficiency
    
    # Features
    supports_streaming: bool = True
    supports_checkpoint: bool = True
    supports_batching: bool = True


@dataclass
class Device:
    """A Probabilistic Compute Unit (PCU) in the system.
    
    Represents a single compute device capable of running THRML sampling jobs.
    Can be a CPU, GPU, or future Extropic hardware.
    """
    
    # Identity
    id: str = field(default_factory=lambda: f"pcu-{uuid.uuid4().hex[:8]}")
    name: str = "Unnamed Device"
    device_type: DeviceType = DeviceType.CPU
    
    # Status
    status: DeviceStatus = DeviceStatus.AVAILABLE
    current_job_id: Optional[str] = None
    
    # Capabilities
    capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)
    
    # Metadata
    registered_at: datetime = field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    
    # Statistics
    jobs_completed: int = 0
    total_samples_generated: int = 0
    total_runtime_seconds: float = 0.0
    
    # Backend-specific info
    backend_info: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """Check if device can accept new jobs."""
        return self.status == DeviceStatus.AVAILABLE
    
    @property
    def is_busy(self) -> bool:
        """Check if device is currently running a job."""
        return self.status == DeviceStatus.BUSY
    
    @property
    def utilization(self) -> float:
        """Memory utilization as a percentage."""
        if self.capabilities.total_memory_gb == 0:
            return 0.0
        used = self.capabilities.total_memory_gb - self.capabilities.available_memory_gb
        return (used / self.capabilities.total_memory_gb) * 100
    
    def assign_job(self, job_id: str) -> None:
        """Assign a job to this device."""
        self.status = DeviceStatus.BUSY
        self.current_job_id = job_id
    
    def release_job(self) -> None:
        """Release the current job."""
        self.status = DeviceStatus.AVAILABLE
        self.current_job_id = None
        self.jobs_completed += 1
    
    def mark_offline(self) -> None:
        """Mark device as offline."""
        self.status = DeviceStatus.OFFLINE
        self.current_job_id = None
    
    def heartbeat(self) -> None:
        """Update last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "device_type": self.device_type.value,
            "status": self.status.value,
            "current_job_id": self.current_job_id,
            "capabilities": {
                "total_memory_gb": self.capabilities.total_memory_gb,
                "available_memory_gb": self.capabilities.available_memory_gb,
                "compute_units": self.capabilities.compute_units,
                "max_nodes": self.capabilities.max_nodes,
                "samples_per_second": self.capabilities.samples_per_second,
            },
            "utilization": self.utilization,
            "jobs_completed": self.jobs_completed,
            "total_samples_generated": self.total_samples_generated,
            "registered_at": self.registered_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
        }


def detect_devices() -> list[Device]:
    """Auto-detect available compute devices.
    
    Returns a list of Device objects representing available PCUs.
    """
    devices = []
    
    # Always add CPU
    import multiprocessing
    cpu_device = Device(
        id="pcu-cpu-0",
        name="CPU",
        device_type=DeviceType.CPU,
        capabilities=DeviceCapabilities(
            compute_units=multiprocessing.cpu_count(),
            max_nodes=100000,
            max_edges=1000000,
        ),
    )
    devices.append(cpu_device)
    
    # Try to detect JAX devices
    try:
        import jax
        jax_devices = jax.devices()
        
        for i, jax_dev in enumerate(jax_devices):
            if jax_dev.platform == "gpu":
                gpu_device = Device(
                    id=f"pcu-gpu-{i}",
                    name=f"GPU {i} ({jax_dev.device_kind})",
                    device_type=DeviceType.GPU_CUDA,
                    capabilities=DeviceCapabilities(
                        compute_units=1,  # Would need CUDA API for actual count
                        max_nodes=1000000,
                        max_edges=10000000,
                        supports_batching=True,
                    ),
                    backend_info={"jax_device": str(jax_dev)},
                )
                devices.append(gpu_device)
            elif jax_dev.platform == "tpu":
                tpu_device = Device(
                    id=f"pcu-tpu-{i}",
                    name=f"TPU {i}",
                    device_type=DeviceType.TPU,
                    capabilities=DeviceCapabilities(
                        max_nodes=10000000,
                        max_edges=100000000,
                    ),
                    backend_info={"jax_device": str(jax_dev)},
                )
                devices.append(tpu_device)
    except ImportError:
        pass
    
    return devices
