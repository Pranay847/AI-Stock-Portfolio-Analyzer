"""Tests for the FastAPI app, with the analyzer and SnapTrade replaced by fakes."""
import logging
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agents.langgraph_workflow as workflow
from api.main import create_app
from api.sessions import COOKIE_NAME
from api.settings import load_settings

SECRET = "s3cr3t-user-secret-value"


# ------------------------------------------------------------------ fakes

class FakeVectorDB:
    def fetch_quote(self, symbol):
        return {"price": 100.0, "change_percent": "2.0", "latest_trading_day": "2026-10-28",
                "market_state": "CLOSED", "source": "yahoo"}


class FakeAnalyzer:
    """Stands in for RobinhoodPortfolioAnalyzer."""
    built = 0

    def __init__(self, gate: threading.Event | None = None, nan: bool = False):
        FakeAnalyzer.built += 1
        self.portfolio = []
        self.last_analysis_notes = []
        self.vector_db = FakeVectorDB()
        self.ai_available = True
        self.ai_backend = "Mistral API"
        self._gate = gate
        self._nan = nan
        self.analyzed = 0  # positions reached, to observe early cancellation

    def analyze_stock(self, symbol, position, *a, **k):
        return {"symbol": symbol, "recommendation": "BUY", "confidence": 72,
                "summary": "fake view", "reasons": ["momentum"], "risks": ["volatility"],
                "analysis_type": "rule_based"}

    def analyze_portfolio(self, progress_callback=None, **kwargs):
        if self._gate:
            self._gate.wait(5)
        out = []
        for i, p in enumerate(self.portfolio, 1):
            self.analyzed += 1
            out.append({"symbol": p["symbol"], "recommendation": "HOLD", "confidence": 60,
                        "summary": f"view on {p['symbol']}", "reasons": ["steady"],
                        "current_price": p["current_price"],
                        "profit_loss_percent": float("nan") if self._nan else p["profit_loss_percent"]})
            if progress_callback:
                progress_callback(i, len(self.portfolio), p["symbol"])
        self.last_analysis_notes = ["1 position skipped"]
        return out


class FakeSnapTrade:
    def __init__(self, personal=False, fail_with: Exception | None = None, positions=None):
        self.is_personal = personal
        self._fail = fail_with
        self._positions = positions
        self.calls = []

    def start_connection(self, redirect_uri=None):
        self.calls.append(("connect", redirect_uri))
        if self._fail:
            raise self._fail
        if self.is_personal:
            return None, None, "https://portal.example/abc"
        return "user-1", SECRET, "https://portal.example/abc"

    def fetch_positions(self, user_id=None, user_secret=None):
        self.calls.append(("fetch", user_id, user_secret))
        if self._fail:
            raise self._fail
        if self._positions is not None:
            return self._positions
        return [{"symbol": "MSFT", "name": "Microsoft", "quantity": 2.0,
                 "average_buy_price": 300.0, "current_price": 330.0, "equity": 660.0,
                 "cost_basis": 600.0, "profit_loss": 60.0, "profit_loss_percent": 10.0}]


OWNER_TOKEN = "correct-horse-battery-staple"


def make_client(tmp_path, snaptrade=None, analyzer_factory=None, static_dir=None,
                owner_token=None):
    settings = replace(load_settings(), static_dir=Path(static_dir or tmp_path / "no-build"),
                       max_concurrent_analyses=2, owner_access_token=owner_token)
    app = create_app(
        settings,
        analyzer_factory=analyzer_factory or (lambda: FakeAnalyzer()),
        snaptrade_factory=lambda s: snaptrade,
    )
    return TestClient(app)


def wait_for(client, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get("/api/portfolio/analysis").json()
        if predicate(state):
            return state
        time.sleep(0.02)
    raise AssertionError(f"timed out; last state: {state}")


@pytest.fixture(autouse=True)
def graph_fails(monkeypatch):
    """Route single-stock analysis through the analyzer, not a real LLM."""
    monkeypatch.setattr(workflow, "run_analysis", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(workflow, "set_vector_db", lambda db: None)


# ------------------------------------------------------------- health etc.

def test_health_is_fast_and_does_not_build_the_analyzer(tmp_path):
    FakeAnalyzer.built = 0
    client = make_client(tmp_path)
    assert client.get("/api/health").json() == {"status": "ok"}
    assert FakeAnalyzer.built == 0


def test_capabilities(tmp_path):
    client = make_client(tmp_path, snaptrade=FakeSnapTrade(personal=True))
    assert client.get("/api/capabilities").json() == {
        "market_data": True, "ai_backend": "Mistral API", "snaptrade": "personal"}


def test_capabilities_when_analyzer_cannot_start(tmp_path):
    client = make_client(tmp_path, analyzer_factory=lambda: None)
    assert client.get("/api/capabilities").json() == {
        "market_data": False, "ai_backend": None, "snaptrade": None}


# ----------------------------------------------------------- stock analysis

def test_analyze_stock_success(tmp_path):
    r = make_client(tmp_path).post("/api/stocks/analyze", json={"query": "apple"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["symbol"] == "AAPL"
    assert body["recommendation"] == "BUY"
    assert body["current_price"] == 100.0
    assert body["reasons"] == ["momentum"]
    assert body["as_of"] == "2026-10-28"


def test_analyze_stock_failure_is_ok_false_not_a_500(tmp_path):
    client = make_client(tmp_path, analyzer_factory=lambda: None)
    body = client.post("/api/stocks/analyze", json={"query": "AAPL"}).json()
    assert body["ok"] is False
    assert body["current_price"] == 0


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x" * 65}, {}])
def test_analyze_stock_validates_input(tmp_path, payload):
    assert make_client(tmp_path).post("/api/stocks/analyze", json=payload).status_code == 422


# ----------------------------------------------------------------- sessions

def test_session_cookie_flags(tmp_path):
    r = make_client(tmp_path).get("/api/portfolio")
    cookie = r.headers["set-cookie"].lower()
    assert cookie.startswith(f"{COOKIE_NAME}=")
    assert "httponly" in cookie
    assert "samesite=lax" in cookie


def test_cookie_is_reused_not_reissued(tmp_path):
    client = make_client(tmp_path)
    client.get("/api/portfolio")
    assert "set-cookie" not in client.get("/api/portfolio").headers


def test_forged_session_id_is_not_adopted(tmp_path):
    client = make_client(tmp_path)
    client.cookies.set(COOKIE_NAME, "attacker-chosen-id")
    r = client.get("/api/portfolio")
    issued = r.cookies.get(COOKIE_NAME)
    assert issued and issued != "attacker-chosen-id"


def test_sessions_are_isolated(tmp_path):
    settings = replace(load_settings(), static_dir=tmp_path / "x")
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(),
                     snaptrade_factory=lambda s: None)
    alice, bob = TestClient(app), TestClient(app)
    alice.post("/api/portfolio/demo")
    assert len(alice.get("/api/portfolio").json()["positions"]) == 4
    assert bob.get("/api/portfolio").json()["positions"] == []


# ---------------------------------------------------------------- portfolio

def test_demo_portfolio_round_trip(tmp_path):
    client = make_client(tmp_path)
    empty = client.get("/api/portfolio").json()
    assert empty["source"] is None and empty["positions"] == [] and empty["summary"] is None

    loaded = client.post("/api/portfolio/demo").json()
    assert loaded["source"] == "demo"
    assert [p["symbol"] for p in loaded["positions"]] == ["AAPL", "GOOGL", "TSLA", "NVDA"]
    assert loaded["summary"]["total_positions"] == 4
    assert loaded["summary"]["best_performer"] == "TSLA"

    cleared = client.delete("/api/portfolio").json()
    assert cleared["source"] is None and cleared["positions"] == []


# ---------------------------------------------------------------- snaptrade

def test_connect_without_snaptrade_configured(tmp_path):
    client = make_client(tmp_path, snaptrade=None)
    assert client.post("/api/snaptrade/connect").status_code == 409
    assert client.get("/api/portfolio").json()["snaptrade"] == {
        "configured": False, "mode": None, "available": False, "owner_unlock_required": False,
        "unavailable_reason": None, "connection_pending": False}


def test_commercial_connect_and_sync_never_expose_the_user_secret(tmp_path):
    snap = FakeSnapTrade(personal=False)
    client = make_client(tmp_path, snaptrade=snap)

    connect = client.post("/api/snaptrade/connect")
    assert connect.json() == {"portal_url": "https://portal.example/abc"}
    pending = client.get("/api/portfolio")
    assert pending.json()["snaptrade"]["connection_pending"] is True

    synced = client.post("/api/snaptrade/sync")
    assert synced.json()["source"] == "snaptrade"
    assert synced.json()["positions"][0]["symbol"] == "MSFT"
    assert synced.json()["snaptrade"]["connection_pending"] is False
    # The server used the secret...
    assert ("fetch", "user-1", SECRET) in snap.calls
    # ...but no response ever carried it.
    for r in (connect, pending, synced):
        assert SECRET not in r.text
        assert SECRET not in str(r.headers)


def test_personal_mode_sends_no_user_credentials(tmp_path):
    snap = FakeSnapTrade(personal=True)
    client = make_client(tmp_path, snaptrade=snap, owner_token=OWNER_TOKEN)
    assert client.post("/api/owner/unlock", json={"token": OWNER_TOKEN}).status_code == 204
    client.post("/api/snaptrade/connect")
    client.post("/api/snaptrade/sync")
    assert ("fetch", None, None) in snap.calls


def test_sync_before_connect_is_rejected(tmp_path):
    client = make_client(tmp_path, snaptrade=FakeSnapTrade())
    assert client.post("/api/snaptrade/sync").status_code == 409


def test_sync_with_no_positions_keeps_the_connection_pending(tmp_path):
    client = make_client(tmp_path, snaptrade=FakeSnapTrade(positions=[]))
    client.post("/api/snaptrade/connect")
    assert client.post("/api/snaptrade/sync").status_code == 404
    assert client.get("/api/portfolio").json()["snaptrade"]["connection_pending"] is True


def test_snaptrade_errors_are_scrubbed_from_response_and_logs(tmp_path, caplog):
    leaky = ConnectionError(
        f"Max retries exceeded with url: /api/v1/accounts?userId=u&userSecret={SECRET}&timestamp=1")
    client = make_client(tmp_path, snaptrade=FakeSnapTrade(fail_with=leaky))
    with caplog.at_level(logging.WARNING, logger="api.main"):
        r = client.post("/api/snaptrade/connect")
    assert r.status_code == 502
    assert SECRET not in r.text
    assert SECRET not in caplog.text
    assert "userSecret=[redacted]" in caplog.text


def test_snaptrade_http_status_is_reported(tmp_path):
    err = Exception("rejected")
    err.status = 403
    client = make_client(tmp_path, snaptrade=FakeSnapTrade(fail_with=err))
    r = client.post("/api/snaptrade/connect")
    assert r.status_code == 502
    assert "HTTP 403" in r.json()["detail"]


# ----------------------------------------------------------- analysis jobs

def test_analysis_requires_a_portfolio(tmp_path):
    assert make_client(tmp_path).post("/api/portfolio/analysis").status_code == 409


def test_analysis_job_lifecycle(tmp_path):
    gate = threading.Event()
    client = make_client(tmp_path, analyzer_factory=lambda: FakeAnalyzer(gate=gate))
    assert client.get("/api/portfolio/analysis").json()["status"] == "idle"
    client.post("/api/portfolio/demo")

    started = client.post("/api/portfolio/analysis")
    assert started.status_code == 202
    assert started.json()["status"] in ("queued", "running")
    assert started.json()["total"] == 4
    # A second start while one is in flight is refused.
    assert client.post("/api/portfolio/analysis").status_code == 409

    gate.set()
    done = wait_for(client, lambda s: s["status"] == "done")
    assert done["completed"] == 4
    assert [r["symbol"] for r in done["results"]] == ["AAPL", "GOOGL", "TSLA", "NVDA"]
    assert done["notes"] == ["1 position skipped"]

    report = client.get("/api/portfolio/analysis/report")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/plain")
    assert "attachment" in report.headers["content-disposition"]
    assert "PORTFOLIO ANALYSIS REPORT" in report.text


def test_each_job_gets_a_private_analyzer(tmp_path):
    built = []

    def factory():
        a = FakeAnalyzer()
        built.append(a)
        return a

    client = make_client(tmp_path, analyzer_factory=factory)
    client.post("/api/stocks/analyze", json={"query": "AAPL"})  # builds the shared one
    shared = built[0]
    client.post("/api/portfolio/demo")
    client.post("/api/portfolio/analysis")
    wait_for(client, lambda s: s["status"] == "done")
    assert len(built) == 2
    assert shared.portfolio == []  # the shared instance never held a portfolio


def test_nan_in_results_does_not_break_the_response(tmp_path):
    client = make_client(tmp_path, analyzer_factory=lambda: FakeAnalyzer(nan=True))
    client.post("/api/portfolio/demo")
    client.post("/api/portfolio/analysis")
    done = wait_for(client, lambda s: s["status"] == "done")
    assert done["results"][0]["profit_loss_percent"] is None
    assert client.get("/api/portfolio/analysis/report").status_code == 200


def test_failed_job_reports_an_error(tmp_path):
    client = make_client(tmp_path, analyzer_factory=lambda: None)
    client.post("/api/portfolio/demo")
    client.post("/api/portfolio/analysis")
    failed = wait_for(client, lambda s: s["status"] == "error")
    assert "failed to initialise" in failed["error"]
    assert client.get("/api/portfolio/analysis/report").status_code == 409


def test_clearing_the_portfolio_drops_the_analysis(tmp_path):
    client = make_client(tmp_path)
    client.post("/api/portfolio/demo")
    client.post("/api/portfolio/analysis")
    wait_for(client, lambda s: s["status"] == "done")
    client.delete("/api/portfolio")
    assert client.get("/api/portfolio/analysis").json()["status"] == "idle"


# ----------------------------------------------------------------- frontend

@pytest.fixture
def build_dir(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>spa</title>")
    (dist / "assets" / "app-abc123.js").write_text("console.log('app')")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside the build")
    return dist


def test_spa_serves_index_assets_and_client_routes(tmp_path, build_dir):
    client = make_client(tmp_path, static_dir=build_dir)
    root = client.get("/")
    assert root.status_code == 200 and "<title>spa</title>" in root.text
    assert root.headers["cache-control"] == "no-cache"
    assert client.get("/assets/app-abc123.js").text == "console.log('app')"
    assert client.get("/favicon.svg").text == "<svg/>"
    assert "<title>spa</title>" in client.get("/portfolio/some/client/route").text


def test_unknown_api_routes_stay_json_404(tmp_path, build_dir):
    r = make_client(tmp_path, static_dir=build_dir).get("/api/does-not-exist")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")


def test_spa_fallback_blocks_path_traversal(tmp_path, build_dir):
    client = make_client(tmp_path, static_dir=build_dir)
    for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"):
        assert "outside the build" not in client.get(path).text


# ------------------------------------------------- personal key owner gate
# A personal SnapTrade key resolves every call to its owner's account, so a
# visitor who could reach connect/sync would be shown the owner's holdings.

def _personal(tmp_path, token=OWNER_TOKEN, snap=None):
    snap = snap or FakeSnapTrade(personal=True)
    return make_client(tmp_path, snaptrade=snap, owner_token=token), snap


@pytest.mark.parametrize("token", [None, "", "too-short"])
def test_personal_key_without_a_usable_owner_token_is_disabled(tmp_path, token):
    client, snap = _personal(tmp_path, token=token)
    state = client.get("/api/portfolio").json()["snaptrade"]
    assert state["available"] is False
    assert state["owner_unlock_required"] is False
    assert "owner token" in state["unavailable_reason"]
    assert client.post("/api/snaptrade/connect").status_code == 409
    assert client.post("/api/snaptrade/sync").status_code == 409
    assert snap.calls == []  # SnapTrade was never contacted


def test_personal_key_requires_unlock_before_connecting(tmp_path):
    client, snap = _personal(tmp_path)
    state = client.get("/api/portfolio").json()["snaptrade"]
    assert state["available"] is False and state["owner_unlock_required"] is True
    assert client.post("/api/snaptrade/connect").status_code == 403
    assert client.post("/api/snaptrade/sync").status_code == 403
    assert snap.calls == []


def test_wrong_owner_token_is_rejected(tmp_path):
    client, snap = _personal(tmp_path)
    assert client.post("/api/owner/unlock", json={"token": "nope-" + OWNER_TOKEN}).status_code == 403
    assert client.post("/api/snaptrade/connect").status_code == 403
    assert snap.calls == []


def test_owner_can_connect_after_unlocking(tmp_path):
    client, snap = _personal(tmp_path)
    assert client.post("/api/owner/unlock", json={"token": OWNER_TOKEN}).status_code == 204
    assert client.get("/api/portfolio").json()["snaptrade"]["available"] is True
    assert client.post("/api/snaptrade/connect").status_code == 200
    synced = client.post("/api/snaptrade/sync").json()
    assert synced["positions"][0]["symbol"] == "MSFT"


def test_an_unlocked_owner_does_not_unlock_other_visitors(tmp_path):
    snap = FakeSnapTrade(personal=True)
    settings = replace(load_settings(), static_dir=tmp_path / "x", owner_access_token=OWNER_TOKEN)
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(),
                     snaptrade_factory=lambda s: snap)
    owner, visitor = TestClient(app), TestClient(app)

    owner.post("/api/owner/unlock", json={"token": OWNER_TOKEN})
    owner.post("/api/snaptrade/connect")
    assert owner.post("/api/snaptrade/sync").status_code == 200

    # The visitor shares the server and the key, but not the owner's session.
    assert visitor.post("/api/snaptrade/connect").status_code == 403
    assert visitor.post("/api/snaptrade/sync").status_code == 403
    assert visitor.get("/api/portfolio").json()["positions"] == []
    assert sum(1 for c in snap.calls if c[0] == "fetch") == 1  # only the owner's


def test_disconnect_signs_the_owner_out(tmp_path):
    client, _ = _personal(tmp_path)
    client.post("/api/owner/unlock", json={"token": OWNER_TOKEN})
    client.delete("/api/portfolio")
    assert client.get("/api/portfolio").json()["snaptrade"]["owner_unlock_required"] is True
    assert client.post("/api/snaptrade/connect").status_code == 403


def test_commercial_key_needs_no_owner_token(tmp_path):
    client = make_client(tmp_path, snaptrade=FakeSnapTrade(personal=False))
    state = client.get("/api/portfolio").json()["snaptrade"]
    assert state["available"] is True and state["owner_unlock_required"] is False
    assert client.post("/api/snaptrade/connect").status_code == 200


def test_owner_token_is_never_echoed(tmp_path):
    client, _ = _personal(tmp_path)
    responses = [
        client.post("/api/owner/unlock", json={"token": OWNER_TOKEN}),
        client.get("/api/portfolio"),
        client.get("/api/capabilities"),
    ]
    for r in responses:
        assert OWNER_TOKEN not in r.text


# -------------------------------------------- fixes from the code review

def test_owner_unlock_rotates_the_session_id(tmp_path):
    client, _ = _personal(tmp_path)
    before = client.get("/api/portfolio").cookies.get(COOKIE_NAME)
    after = client.post("/api/owner/unlock", json={"token": OWNER_TOKEN}).cookies.get(COOKIE_NAME)
    assert before and after and after != before
    client.cookies.clear()
    client.cookies.set(COOKIE_NAME, after)
    assert client.get("/api/portfolio").json()["snaptrade"]["available"] is True


def test_a_planted_session_id_does_not_inherit_the_owner_unlock(tmp_path):
    """Session fixation: the attacker gets a real id, plants it in the owner's
    browser, and waits for the owner to unlock on it."""
    snap = FakeSnapTrade(personal=True)
    settings = replace(load_settings(), static_dir=tmp_path / "x", owner_access_token=OWNER_TOKEN)
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(), snaptrade_factory=lambda s: snap)
    attacker, owner = TestClient(app), TestClient(app)

    planted = attacker.get("/api/portfolio").cookies.get(COOKIE_NAME)
    owner.cookies.set(COOKIE_NAME, planted)
    unlocked = owner.post("/api/owner/unlock", json={"token": OWNER_TOKEN})
    assert unlocked.status_code == 204
    owner.cookies.clear()
    owner.cookies.set(COOKIE_NAME, unlocked.cookies.get(COOKIE_NAME))
    owner.post("/api/snaptrade/connect")
    assert owner.post("/api/snaptrade/sync").status_code == 200

    # The attacker still holds the planted id, which no longer exists.
    assert attacker.post("/api/snaptrade/connect").status_code == 403
    assert attacker.post("/api/snaptrade/sync").status_code == 403
    assert attacker.get("/api/portfolio").json()["positions"] == []
    assert sum(1 for c in snap.calls if c[0] == "fetch") == 1


def test_https_uses_a_host_prefixed_secure_cookie(tmp_path):
    settings = replace(load_settings(), static_dir=tmp_path / "x")
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(), snaptrade_factory=lambda s: None)
    cookie = TestClient(app, base_url="https://testserver").get("/api/portfolio").headers["set-cookie"]
    low = cookie.lower()
    assert cookie.startswith("__Host-sid=")
    assert "secure" in low and "httponly" in low and "path=/" in low and "domain=" not in low


def test_a_cookie_issued_over_http_is_not_honoured_over_https(tmp_path):
    settings = replace(load_settings(), static_dir=tmp_path / "x")
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(), snaptrade_factory=lambda s: None)
    http = TestClient(app, base_url="http://testserver")
    http.post("/api/portfolio/demo")
    https = TestClient(app, base_url="https://testserver")
    https.cookies.set(COOKIE_NAME, http.cookies.get(COOKIE_NAME))
    r = https.get("/api/portfolio")
    assert r.json()["positions"] == []  # a fresh session, not the non-Secure one
    assert r.headers["set-cookie"].startswith("__Host-sid=")


@pytest.mark.parametrize("action", ["disconnect", "reload_demo", "resync"])
def test_replacing_the_portfolio_cancels_the_running_analysis(tmp_path, action):
    gate = threading.Event()
    built = []

    def factory():
        built.append(FakeAnalyzer(gate=gate))
        return built[-1]

    snap = FakeSnapTrade(personal=False)
    client = make_client(tmp_path, analyzer_factory=factory, snaptrade=snap)
    if action == "resync":
        client.post("/api/snaptrade/connect")
        client.post("/api/snaptrade/sync")
    else:
        client.post("/api/portfolio/demo")
    client.post("/api/portfolio/analysis")
    wait_for(client, lambda s: s["status"] == "running")

    {
        "disconnect": lambda: client.delete("/api/portfolio"),
        "reload_demo": lambda: client.post("/api/portfolio/demo"),
        "resync": lambda: client.post("/api/snaptrade/sync"),
    }[action]()
    gate.set()  # let the job reach its first progress check

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and built[0].analyzed == 0:
        time.sleep(0.02)
    time.sleep(0.2)
    assert built[0].analyzed == 1, "the cancelled job kept analyzing positions"
    assert client.get("/api/portfolio/analysis").json()["status"] == "idle"


def test_the_analysis_queue_is_bounded(tmp_path):
    gate = threading.Event()
    settings = replace(load_settings(), static_dir=tmp_path / "x",
                       max_concurrent_analyses=1, max_pending_analyses=2)
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(gate=gate),
                     snaptrade_factory=lambda s: None)
    clients = [TestClient(app) for _ in range(3)]
    for c in clients:
        c.post("/api/portfolio/demo")

    assert clients[0].post("/api/portfolio/analysis").status_code == 202
    assert clients[1].post("/api/portfolio/analysis").status_code == 202
    refused = clients[2].post("/api/portfolio/analysis")
    assert refused.status_code == 503
    assert refused.headers["retry-after"] == "30"

    gate.set()
    wait_for(clients[0], lambda s: s["status"] == "done")
    wait_for(clients[1], lambda s: s["status"] == "done")
    # Finished jobs free their slots.
    assert clients[2].post("/api/portfolio/analysis").status_code == 202


def test_cancelled_jobs_free_their_slot(tmp_path):
    gate = threading.Event()
    settings = replace(load_settings(), static_dir=tmp_path / "x",
                       max_concurrent_analyses=1, max_pending_analyses=1)
    app = create_app(settings, analyzer_factory=lambda: FakeAnalyzer(gate=gate),
                     snaptrade_factory=lambda s: None)
    client = TestClient(app)
    client.post("/api/portfolio/demo")
    assert client.post("/api/portfolio/analysis").status_code == 202
    client.delete("/api/portfolio")  # cancels
    gate.set()
    client.post("/api/portfolio/demo")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        r = client.post("/api/portfolio/analysis")
        if r.status_code == 202:
            break
        time.sleep(0.05)
    assert r.status_code == 202
