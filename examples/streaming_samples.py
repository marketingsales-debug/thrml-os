#!/usr/bin/env python3
"""Streaming samples example.

Demonstrates how to consume samples as they're generated,
without waiting for the entire job to complete.
"""

import time
from thrml_os.client import THRMLClient
from thrml_os.models.job import JobPriority


class MockModel:
    """Mock model for demonstration."""
    def __init__(self, n_nodes=10):
        self.nodes = list(range(n_nodes))
        self.edges = []


def main():
    """Run streaming example."""
    print("=" * 60)
    print("THRML-OS Streaming Samples Example")
    print("=" * 60)
    
    client = THRMLClient()
    model = MockModel(n_nodes=8)
    
    print("\n1. Starting streaming job...")
    
    total_samples = 0
    batch_count = 0
    
    # Stream samples as they're generated
    for batch in client.stream(
        model=model,
        n_samples=200,
        n_warmup=20,
        priority=JobPriority.HIGH,
    ):
        batch_count += 1
        total_samples += batch.n_samples
        
        print(f"   Batch {batch_count}: "
              f"{batch.n_samples} samples, "
              f"energy range [{batch.energies.min():.2f}, {batch.energies.max():.2f}], "
              f"total: {total_samples}")
    
    print(f"\n2. Summary:")
    print(f"   Total batches: {batch_count}")
    print(f"   Total samples: {total_samples}")
    
    client.stop()
    
    print("\n" + "=" * 60)
    print("Streaming example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
