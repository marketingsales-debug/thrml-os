#!/usr/bin/env python3
"""Basic THRML-OS usage example.

This demonstrates the core features of THRML-OS:
- Creating sampling jobs
- Submitting to the scheduler
- Monitoring progress
- Collecting results
"""

import time
import jax
import jax.numpy as jnp

from thrml_os.client import THRMLClient
from thrml_os.models.job import JobPriority


def main():
    """Run basic example."""
    print("=" * 60)
    print("THRML-OS Basic Example")
    print("=" * 60)
    
    # Create a simple model
    # In real usage, this would be a THRML EBM or PGM
    class SimpleModel:
        """A simple test model."""
        def __init__(self, n_nodes=10):
            self.nodes = list(range(n_nodes))
            self.edges = [(i, i+1) for i in range(n_nodes - 1)]
    
    # Initialize client
    client = THRMLClient()
    
    print("\n1. Listing available devices...")
    devices = client.list_devices()
    for device in devices:
        print(f"   - {device.name} ({device.device_type.value})")
    
    # Create model
    model = SimpleModel(n_nodes=20)
    print(f"\n2. Created model with {len(model.nodes)} nodes, {len(model.edges)} edges")
    
    # Submit a job
    print("\n3. Submitting sampling job...")
    job = client.submit(
        model=model,
        n_samples=100,
        n_warmup=20,
        priority=JobPriority.NORMAL,
    )
    print(f"   Job ID: {job.id}")
    
    # Monitor progress
    print("\n4. Monitoring job progress...")
    while not job.is_terminal:
        time.sleep(0.1)
        job = client.get_job(job.id)
        print(f"   Status: {job.status.value}, Progress: {job.progress:.1f}%")
        if job.is_terminal:
            break
    
    # Get results
    print("\n5. Results:")
    if job.samples is not None:
        print(f"   Samples shape: {job.samples.shape}")
        print(f"   Sample mean: {jnp.mean(job.samples):.4f}")
        print(f"   Sample std: {jnp.std(job.samples):.4f}")
    
    # Cleanup
    client.stop()
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
