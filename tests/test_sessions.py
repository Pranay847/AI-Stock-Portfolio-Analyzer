import pytest

import api.sessions as sessions
from api.sessions import Session, SessionStore


class _Clock:
    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(sessions.time, "monotonic", c)
    return c


def test_unknown_or_missing_id_gets_fresh_random_id(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    s1, created1 = store.get_or_create(None)
    s2, created2 = store.get_or_create("attacker-chosen-id")
    assert created1 and created2
    assert s2.id != "attacker-chosen-id"
    assert s1.id != s2.id
    assert len(store) == 2


def test_known_id_returns_same_session(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    s, _ = store.get_or_create(None)
    again, created = store.get_or_create(s.id)
    assert again is s and not created


def test_session_expires_after_ttl_but_not_at_ttl(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    s, _ = store.get_or_create(None)
    clock.now += 60  # exactly at TTL: still live
    assert store.get_or_create(s.id) == (s, False)
    clock.now += 61
    fresh, created = store.get_or_create(s.id)
    assert created and fresh.id != s.id
    assert len(store) == 1


def test_lru_eviction_keeps_recently_touched(clock):
    store = SessionStore(ttl_seconds=600, max_sessions=2)
    a, _ = store.get_or_create(None)
    b, _ = store.get_or_create(None)
    store.get_or_create(a.id)  # a becomes most recent
    c, _ = store.get_or_create(None)  # evicts b
    assert len(store) == 2
    assert store.get_or_create(a.id)[1] is False
    assert store.get_or_create(c.id)[1] is False
    assert store.get_or_create(b.id)[0].id != b.id


def test_rotate_retires_old_id_and_keeps_state(clock):
    store = SessionStore(ttl_seconds=60, max_sessions=10)
    s, _ = store.get_or_create(None)
    s.owner_unlocked = True
    old_id = s.id
    new_id = store.rotate(s)
    assert new_id != old_id and s.id == new_id
    assert len(store) == 1
    same, created = store.get_or_create(new_id)
    assert same is s and same.owner_unlocked and not created
    assert store.get_or_create(old_id)[0] is not s


def test_clear_portfolio_cancels_analysis_and_wipes_secrets():
    class FakeJob:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    job = FakeJob()
    s = Session(id="x", portfolio_source="snaptrade", positions=[{"symbol": "AAPL"}],
                snaptrade_user_id="u", snaptrade_user_secret="secret",
                snaptrade_portal_url="https://example", analysis=job)
    s.owner_unlocked = True
    s.clear_portfolio()
    assert job.cancelled
    assert s.analysis is None and s.positions == [] and s.portfolio_source is None
    assert s.snaptrade_user_secret is None and s.snaptrade_user_id is None
    assert s.snaptrade_portal_url is None
    assert s.owner_unlocked  # unlock is a session privilege, not portfolio state


def test_cancel_analysis_without_job_is_noop():
    s = Session(id="x")
    s.cancel_analysis()
    assert s.analysis is None
