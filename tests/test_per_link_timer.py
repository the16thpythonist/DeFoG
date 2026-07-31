"""PerLinkTimer must ignore restored elapsed time; stock Timer must not."""
import pytest
import pytorch_lightning as pl
from defog.core import PerLinkTimer

STATE = {"time_elapsed": {"train": 37562.6, "sanity_check": 0,
                          "validate": 3.3, "test": 0, "predict": 0}}


def test_stock_timer_restores_elapsed_this_is_the_bug():
    """Baseline: this is why a chained run silently stops making progress."""
    t = pl.callbacks.Timer(duration={"hours": 10, "minutes": 30})
    t.load_state_dict(dict(STATE))
    assert t.time_elapsed("train") > 37000        # budget already spent
    assert t.time_elapsed("train") > 10 * 3600    # ...essentially all of it


def test_per_link_timer_ignores_restored_elapsed():
    t = PerLinkTimer(duration={"hours": 10, "minutes": 30})
    t.load_state_dict(dict(STATE))
    assert t.time_elapsed("train") == 0.0, "restored time must not carry over"


def test_per_link_timer_still_reports_its_own_state():
    t = PerLinkTimer(duration={"hours": 1})
    assert "time_elapsed" in t.state_dict()


def test_per_link_timer_is_a_timer():
    """Must remain drop-in for Trainer(callbacks=[...]) usage."""
    assert isinstance(PerLinkTimer(duration={"hours": 1}), pl.callbacks.Timer)
