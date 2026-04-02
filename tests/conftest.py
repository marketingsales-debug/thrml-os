"""Pytest configuration and fixtures."""

import pytest
import jax
import jax.numpy as jnp

from thrml_os.hal.simulator import SimulatorBackend
from thrml_os.scheduler.scheduler import Scheduler


@pytest.fixture
def rng_key():
    """Provide a JAX PRNG key."""
    return jax.random.key(42)


@pytest.fixture
def simulator_backend():
    """Provide an initialized simulator backend."""
    backend = SimulatorBackend(samples_per_second=100000)
    backend.initialize()
    yield backend
    backend.shutdown()


@pytest.fixture
def scheduler(simulator_backend):
    """Provide a running scheduler."""
    sched = Scheduler(backend=simulator_backend)
    sched.start()
    yield sched
    sched.stop(wait=False)


@pytest.fixture
def mock_model():
    """Provide a simple mock model."""
    class MockModel:
        def __init__(self, n_nodes=10):
            self.nodes = list(range(n_nodes))
            self.edges = [(i, i+1) for i in range(n_nodes - 1)]
    
    return MockModel()


@pytest.fixture
def ising_model():
    """Provide an Ising-like mock model."""
    class IsingModel:
        def __init__(self, size=4):
            self.size = size
            self.nodes = list(range(size * size))
            # 2D lattice edges
            self.edges = []
            for i in range(size):
                for j in range(size):
                    node = i * size + j
                    if j < size - 1:
                        self.edges.append((node, node + 1))
                    if i < size - 1:
                        self.edges.append((node, node + size))
    
    return IsingModel()
