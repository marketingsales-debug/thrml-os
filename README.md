# THRML-OS: Thermodynamic Computing Operating System

<div align="center">
  <h3>Runtime for Probabilistic Graphical Models on Extropic Hardware</h3>
</div>

---

THRML-OS is an operating system layer for probabilistic/thermodynamic computing. It provides job scheduling, resource management, and hardware abstraction for running [THRML](https://github.com/extropic-ai/thrml) workloads efficiently.

## Features

- **Hardware Abstraction Layer (HAL)** - Unified interface for JAX/GPU, JAX/CPU, and future Extropic hardware
- **Job Scheduler** - Priority-based scheduling with preemption at Gibbs sweep boundaries
- **Resource Management** - PCU allocation, memory management, temperature control
- **Checkpoint/Resume** - Fault-tolerant execution with incremental snapshots
- **CLI & SDK** - Easy job submission and monitoring

## Installation

```bash
pip install thrml-os
```

For GPU support:
```bash
pip install thrml-os[gpu]
```

## Quick Start

### Submit a sampling job

```python
from thrml_os import THRMLClient, SamplingJob
from thrml import SpinNode, Block
from thrml.models import IsingEBM

# Define your model
nodes = [SpinNode() for _ in range(100)]
edges = [(nodes[i], nodes[i+1]) for i in range(99)]
model = IsingEBM(nodes, edges, biases, weights, beta)

# Create a job
job = SamplingJob(
    model=model,
    n_samples=1000,
    n_warmup=100,
    priority="normal"
)

# Submit to THRML-OS
client = THRMLClient()
result = client.submit(job)
samples = result.samples
```

### CLI Usage

```bash
# List available devices
thrml-os devices

# Submit a job
thrml-os submit my_model.py --samples 1000

# Check job status
thrml-os status job-abc123

# Stream samples
thrml-os logs job-abc123 --follow
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Space                          │
│   Python SDK  │  CLI Tools  │  REST API  │  Jupyter     │
├─────────────────────────────────────────────────────────┤
│                    Runtime Layer                         │
│   Scheduler  │  Graph Compiler  │  Checkpoint Engine    │
├─────────────────────────────────────────────────────────┤
│                   Resource Layer                         │
│   PCU Allocator  │  Memory Manager  │  Temperature Ctrl │
├─────────────────────────────────────────────────────────┤
│              Hardware Abstraction Layer                  │
│   JAX/GPU  │  JAX/CPU  │  Simulator  │  Extropic HW    │
└─────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone the repository
git clone https://github.com/thrml-os/thrml-os
cd thrml-os

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests
```

## License

Apache 2.0

## Acknowledgments

- [Extropic AI](https://extropic.ai) for THRML and the thermodynamic computing vision
- [JAX](https://jax.readthedocs.io) team for the amazing array computing library
