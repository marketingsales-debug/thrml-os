"""Tests for the Hardware Abstraction Layer."""

import pytest
import jax
import jax.numpy as jnp

from thrml_os.hal.backend import Backend, BackendType, BackendCapabilities, ExecutionContext
from thrml_os.hal.simulator import SimulatorBackend
from thrml_os.hal.registry import BackendRegistry, get_default_backend
from thrml_os.models.schedule import SamplingSchedule


class TestSimulatorBackend:
    """Tests for SimulatorBackend."""
    
    def test_create_backend(self):
        """Test backend creation."""
        backend = SimulatorBackend()
        
        assert backend.backend_type == BackendType.SIMULATOR
        assert backend.capabilities.supports_streaming
    
    def test_initialize_shutdown(self):
        """Test backend lifecycle."""
        backend = SimulatorBackend()
        
        backend.initialize()
        assert backend._initialized
        
        backend.shutdown()
        assert not backend._initialized
    
    def test_compile_model(self):
        """Test model compilation."""
        backend = SimulatorBackend()
        backend.initialize()
        
        class MockModel:
            def __init__(self):
                self.nodes = list(range(10))
                self.edges = [(i, i+1) for i in range(9)]
        
        model = MockModel()
        compiled = backend.compile_model(model)
        
        assert compiled["n_nodes"] == 10
        assert compiled["n_edges"] == 9
        
        backend.shutdown()
    
    def test_sample(self):
        """Test sampling execution."""
        backend = SimulatorBackend(samples_per_second=10000)
        backend.initialize()
        
        class MockModel:
            def __init__(self):
                self.nodes = list(range(5))
                self.edges = []
        
        compiled = backend.compile_model(MockModel())
        schedule = SamplingSchedule(n_warmup=10, n_samples=50, steps_per_sample=1)
        
        key = jax.random.key(42)
        context = ExecutionContext(key=key)
        
        total_samples = 0
        for batch in backend.sample(compiled, schedule, None, context):
            total_samples += batch.n_samples
            assert batch.samples.shape[1] == 5  # n_nodes
            assert batch.energies is not None
        
        assert total_samples == 50
        
        backend.shutdown()
    
    def test_sample_abort(self):
        """Test abort functionality."""
        import threading
        
        backend = SimulatorBackend(samples_per_second=100)
        backend.initialize()
        
        class MockModel:
            nodes = list(range(5))
            edges = []
        
        compiled = backend.compile_model(MockModel())
        schedule = SamplingSchedule(n_warmup=0, n_samples=1000, steps_per_sample=1)
        
        abort = threading.Event()
        key = jax.random.key(42)
        context = ExecutionContext(key=key, should_abort=lambda: abort.is_set())
        
        # Set abort after first batch
        samples_collected = 0
        for batch in backend.sample(compiled, schedule, None, context):
            samples_collected += batch.n_samples
            abort.set()  # Abort after first batch
        
        # Should have stopped early
        assert samples_collected < 1000
        
        backend.shutdown()
    
    def test_checkpoint_restore(self):
        """Test checkpoint serialization."""
        backend = SimulatorBackend()
        backend.initialize()
        
        state = {"spins": jnp.array([1, -1, 1, -1]), "sweep": 50}
        
        checkpoint = backend.checkpoint(state)
        assert isinstance(checkpoint, bytes)
        
        restored = backend.restore(checkpoint)
        assert jnp.array_equal(restored["spins"], state["spins"])
        assert restored["sweep"] == 50
        
        backend.shutdown()


class TestBackendRegistry:
    """Tests for BackendRegistry."""
    
    def test_register_backend(self):
        """Test backend registration."""
        # Clear registry first
        BackendRegistry._backends.clear()
        BackendRegistry._instances.clear()
        
        BackendRegistry.register(BackendType.SIMULATOR, SimulatorBackend)
        
        assert BackendType.SIMULATOR in BackendRegistry.list_available()
    
    def test_get_backend(self):
        """Test backend retrieval."""
        BackendRegistry.register(BackendType.SIMULATOR, SimulatorBackend)
        
        backend = BackendRegistry.get(BackendType.SIMULATOR)
        
        assert backend is not None
        assert backend.backend_type == BackendType.SIMULATOR
    
    def test_get_default_backend(self):
        """Test default backend selection."""
        backend = get_default_backend()
        
        assert backend is not None
        assert isinstance(backend, Backend)
    
    def test_shutdown_all(self):
        """Test shutting down all backends."""
        BackendRegistry.register(BackendType.SIMULATOR, SimulatorBackend)
        backend = BackendRegistry.get(BackendType.SIMULATOR)
        
        BackendRegistry.shutdown_all()
        
        assert len(BackendRegistry._instances) == 0
