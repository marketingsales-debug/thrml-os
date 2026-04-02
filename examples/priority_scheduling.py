#!/usr/bin/env python3
"""Priority scheduling example.

Demonstrates how THRML-OS handles jobs with different priorities,
including preemption of low-priority jobs.
"""

import time
from thrml_os.client import THRMLClient
from thrml_os.models.job import JobPriority


class MockModel:
    """Mock model for demonstration."""
    def __init__(self, n_nodes=10):
        self.nodes = list(range(n_nodes))
        self.edges = [(i, i+1) for i in range(n_nodes - 1)]


def main():
    """Run priority scheduling example."""
    print("=" * 60)
    print("THRML-OS Priority Scheduling Example")
    print("=" * 60)
    
    client = THRMLClient()
    model = MockModel(n_nodes=5)
    
    # Submit jobs with different priorities
    print("\n1. Submitting jobs with different priorities...")
    
    jobs = []
    
    # Batch job (lowest priority)
    batch_job = client.submit(
        model=model,
        n_samples=500,
        priority=JobPriority.BATCH,
    )
    jobs.append(batch_job)
    print(f"   Submitted BATCH job: {batch_job.id}")
    
    # Low priority job
    low_job = client.submit(
        model=model,
        n_samples=500,
        priority=JobPriority.LOW,
    )
    jobs.append(low_job)
    print(f"   Submitted LOW job: {low_job.id}")
    
    # Normal priority job
    normal_job = client.submit(
        model=model,
        n_samples=500,
        priority=JobPriority.NORMAL,
    )
    jobs.append(normal_job)
    print(f"   Submitted NORMAL job: {normal_job.id}")
    
    # High priority job
    high_job = client.submit(
        model=model,
        n_samples=500,
        priority=JobPriority.HIGH,
    )
    jobs.append(high_job)
    print(f"   Submitted HIGH job: {high_job.id}")
    
    # Monitor completion order
    print("\n2. Monitoring job completion order...")
    completed_order = []
    
    while len(completed_order) < len(jobs):
        time.sleep(0.05)
        for job in jobs:
            if job.id not in completed_order:
                updated = client.get_job(job.id)
                if updated.is_terminal:
                    completed_order.append(job.id)
                    print(f"   Completed: {job.id} ({job.priority.name})")
    
    # Analysis
    print("\n3. Analysis:")
    print(f"   Expected order: HIGH → NORMAL → LOW → BATCH")
    print(f"   Actual order:")
    for i, job_id in enumerate(completed_order, 1):
        job = next(j for j in jobs if j.id == job_id)
        print(f"      {i}. {job.priority.name}")
    
    client.stop()
    
    print("\n" + "=" * 60)
    print("Priority scheduling example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
