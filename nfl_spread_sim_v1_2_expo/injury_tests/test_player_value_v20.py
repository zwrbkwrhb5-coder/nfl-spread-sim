import pandas as pd
from score_model.player_value_injury_v20 import starter_score_from_depth

def test_starter_weight_ordering():
    assert starter_score_from_depth("starter") > starter_score_from_depth("2")
    assert starter_score_from_depth("2") > starter_score_from_depth("3")
