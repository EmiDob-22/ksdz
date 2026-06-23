import numpy as np
import pytest

from omega_16d_quantum import Omega16D_Quantum_System


@pytest.fixture
def system():
    np.random.seed(42)
    return Omega16D_Quantum_System()


# --- P4: omega_16d determinism -------------------------------------------
# Omega16D_Quantum_System uses unseeded np.random internally, so these
# tests seed the global RNG around each call rather than asserting
# determinism is a property of the class itself. If the system is meant
# to be reproducible by contract, seeding should move into __init__/evolve.

def test_evolve_produces_expected_shape(system):
    system.evolve(steps=50)
    assert system.history.shape == (50, system.dims)


def test_evolve_is_reproducible_when_globally_seeded():
    np.random.seed(7)
    system_a = Omega16D_Quantum_System()
    system_a.evolve(steps=50)

    np.random.seed(7)
    system_b = Omega16D_Quantum_System()
    system_b.evolve(steps=50)

    np.testing.assert_array_equal(system_a.history, system_b.history)


# --- save_snapshot ---------------------------------------------------------

def test_save_snapshot_noop_without_history(system, tmp_path):
    target = tmp_path / "no_history.ksdz"
    system.save_snapshot(str(target))

    assert not target.exists()


def test_save_snapshot_writes_nonempty_compressed_file(system, tmp_path):
    system.evolve(steps=50)
    target = tmp_path / "trajectory.ksdz"
    system.save_snapshot(str(target))

    assert target.exists()
    raw_size = system.history.astype(np.float32).nbytes
    compressed_size = target.stat().st_size
    assert 0 < compressed_size < raw_size
