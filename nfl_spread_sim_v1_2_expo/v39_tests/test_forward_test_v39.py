from score_model.forward_test_v39 import (
    american_break_even,
    _point_clv,
)


def test_break_even():
    assert abs(american_break_even(-110) - 110/210) < 1e-12
    assert abs(american_break_even(100) - 0.5) < 1e-12


def test_spread_point_clv():
    assert _point_clv(13.5, 12.5, "spread", "MIA") == 1.0
    assert _point_clv(-2.5, -3.0, "spread", "PHI") == 0.5


def test_total_point_clv():
    assert _point_clv(50.5, 52.0, "total", "OVER") == 1.5
    assert _point_clv(50.5, 49.0, "total", "UNDER") == 1.5
