from src.sim_engine import american_break_even, simulate_spread
import numpy as np

def test_break_even_minus_110():
    assert abs(american_break_even(-110) - 0.5238095) < 1e-5

def test_sim_shape():
    residuals = np.linspace(-20, 20, 1000)
    r = simulate_spread(3.0, residuals, home_spread=-2.5, n=10000)
    assert 0 <= r["cover_probability"] <= 1
    assert r["simulations"] == 10000
