from pathlib import Path


def test_close_requires_game_id():
    text = Path("score_model/forward_test_v39.py").read_text()
    assert '"--game-id"' in text
    assert 'action="append"' in text
    assert "required=True" in text
    assert "requested_games" in text
