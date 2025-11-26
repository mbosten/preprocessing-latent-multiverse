from alphacomplexbenchmarking.features.simulate_data import generate_simulation_matrix
import numpy as np


def test_sim_shape_and_seed():
    a = generate_simulation_matrix(3, 4, seed=1)
    b = generate_simulation_matrix(3, 4, seed=1)
    assert a.shape == (3, 4)
    assert np.allclose(a, b)