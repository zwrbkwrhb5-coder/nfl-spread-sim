from score_model.live_first_bet_v23 import american_break_even
def test_minus_110_break_even():
    assert abs(american_break_even(-110)-0.5238095238) < 1e-8
def test_plus_money_break_even():
    assert abs(american_break_even(120)-100/220) < 1e-8
