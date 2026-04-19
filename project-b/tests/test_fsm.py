from presenced.fsm import PresenceFSM, State


def test_present_stays_present():
    f = PresenceFSM(grace_period_s=30.0, now=0.0)
    assert f.observe(True, now=1.0) is None
    assert f.state == State.PRESENT


def test_loss_enters_grace_then_away():
    f = PresenceFSM(grace_period_s=30.0, now=0.0)
    tr = f.observe(False, now=1.0)
    assert tr is not None and tr.to == State.AWAY_GRACE
    assert f.observe(False, now=20.0) is None
    tr2 = f.observe(False, now=31.5)
    assert tr2 is not None and tr2.to == State.AWAY


def test_return_to_present_cancels_grace():
    f = PresenceFSM(grace_period_s=30.0, now=0.0)
    f.observe(False, now=1.0)
    assert f.state == State.AWAY_GRACE
    tr = f.observe(True, now=5.0)
    assert tr is not None and tr.to == State.PRESENT
    assert f.state == State.PRESENT


def test_recover_from_away():
    f = PresenceFSM(grace_period_s=10.0, now=0.0)
    f.observe(False, now=1.0)
    f.observe(False, now=12.0)
    assert f.state == State.AWAY
    tr = f.observe(True, now=13.0)
    assert tr is not None and tr.frm == State.AWAY and tr.to == State.PRESENT
