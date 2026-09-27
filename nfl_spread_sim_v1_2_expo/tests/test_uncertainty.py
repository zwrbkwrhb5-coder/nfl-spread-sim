import numpy as np
from src.uncertainty import expected_abs_to_sigma, scale_residuals_for_game

def test_expected_abs_to_sigma():
    sigma = expected_abs_to_sigma(8.0)
    assert sigma > 8.0

def test_residual_scaling():
    r = np.linspace(-10,10,1000)
    out = scale_residuals_for_game(r, target_sigma=20.0)
    assert 18.0 < np.std(out, ddof=1) < 22.0
