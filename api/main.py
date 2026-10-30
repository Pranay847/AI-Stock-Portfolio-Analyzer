"""FastAPI application: the JSON API under /api, and the built React app at /.

Run locally:   uvicorn api.main:app --reload
In production: one container serves both, so the browser only ever talks to a
single origin -- no CORS, and the session cookie needs no cross-site handling.
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from api.jobs import AnalysisRunner
from api.schemas import (
    AnalysisJobState,
    AnalyzeStockRequest,
    Capabilities,
    ConnectResponse,
    Health,
    OwnerUnlockRequest,
    PortfolioState,
    PortfolioSummary,
    Position,
    PositionAnalysis,
    SnapTradeState,
    StockAnalysis,
)
from api.serialize import jsonable
from api.sessions import COOKIE_NAME, Session, SessionStore
from api.settings import REPO_ROOT, Settings, load_settings
from services.owner_access import token_problem, verify_owner_token
from services.redact import scrub

logger = logging.getLogger(__name__)

_POSITION_NUMBERS = ("quantity", "average_buy_price", "current_price", "equity",
                     "cost_basis", "profit_loss", "profit_loss_percent")


# ---------------------------------------------------------------- factories

def default_analyzer_factory():
    """A new RobinhoodPortfolioAnalyzer, or None if it cannot be built."""
    try:
        from robinhood_portfolio_analyzer import RobinhoodPortfolioAnalyzer
        return RobinhoodPortfolioAnalyzer(vector_db_path=str(REPO_ROOT / "portfolio_vector_db"))
    except Exception:
        logger.exception("analyzer failed to initialise")
        return None


def default_snaptrade_factory(settings: Settings):
    """A SnapTradeConnector, or None if brokerage connection is not configured."""
    if not settings.snaptrade_configured:
        return None
    try:
        from snaptrade_connector import SNAPTRADE_SDK_AVAILABLE, SnapTradeConnector
    except ImportError:
        return None
    if not SNAPTRADE_SDK_AVAILABLE:
        return None
    return SnapTradeConnector(
        client_id=settings.snaptrade_client_id,
        consumer_key=settings.snaptrade_consumer_key,
        auth_mode=settings.snaptrade_auth_mode,
    )


class AppState:
    """Process-wide objects, built once and shared across requests."""

    def __init__(self, settings: Settings, analyzer_factory: Callable[[], Any],
                 snaptrade_factory: Callable[[Settings], Any]):
        self.settings = settings
        self.sessions = SessionStore(settings.session_ttl_seconds, settings.max_sessions)
        self.runner = AnalysisRunner(settings.max_concurrent_analyses, analyzer_factory)
        self._analyzer_factory = analyzer_factory
        self._shared_analyzer = None
        self._analyzer_lock = threading.Lock()
        self.snaptrade = snaptrade_factory(settings)

    def shared_analyzer(self):
        """The analyzer for stateless work (single-stock lookups).

        Shared across sessions, so nothing may store a portfolio on it --
        portfolio jobs build their own. A failed build is retried next time
        rather than cached as None.
        """
        with self._analyzer_lock:
            if self._shared_analyzer is None:
                self._shared_analyzer = self._analyzer_factory()
            return self._shared_analyzer


# ------------------------------------------------------------- dependencies

def app_state(request: Request) -> AppState:
    return request.app.state.app_state


def current_session(request: Request, response: Response,
                    state: AppState = Depends(app_state)) -> Session:
    session, created = state.sessions.get_or_create(request.cookies.get(COOKIE_NAME))
    if created:
        # A browser-session cookie; the server expires idle sessions itself.
        response.set_cookie(
            COOKIE_NAME, session.id, httponly=True, samesite="lax", path="/",
            secure=request.url.scheme == "https",
        )
    return session


# --------------------------------------------------------------- converters

def _strings(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value if v is not None]


def _number(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number and abs(number) != float("inf") else default


def to_stock_analysis(result: dict) -> StockAnalysis:
    r = jsonable(result)
    xgb_conf = r.get("xgboost_confidence")
    return StockAnalysis(
        ok=not r.get("data_fetch_failed", False),
        symbol=str(r.get("symbol", "")),
        current_price=_number(r.get("current_price")),
        buy_price=_number(r.get("buy_price")),
        sell_price=_number(r.get("sell_price")),
        recommendation=str(r.get("recommendation") or "HOLD"),
        confidence=_number(r.get("confidence")),
        holding_period=str(r.get("holding_period") or "N/A"),
        summary=str(r.get("summary") or ""),
        reasons=_strings(r.get("reasons")),
        risks=_strings(r.get("risks")),
        xgboost_signal=r.get("xgboost_signal") or None,
        xgboost_confidence=None if xgb_conf is None else _number(xgb_conf),
        analysis_type=str(r.get("analysis_type") or "unknown"),
        ai_model=str(r.get("ai_model") or "Unknown"),
        as_of=r.get("as_of") or None,
        market_state=r.get("market_state") or None,
        quote_source=r.get("quote_source") or None,
    )


def to_position(p: dict) -> Position:
    p = jsonable(p)
    symbol = str(p.get("symbol", ""))
    return Position(
        symbol=symbol,
        name=str(p.get("name") or symbol),
        **{k: _number(p.get(k)) for k in _POSITION_NUMBERS},
    )


def to_position_analysis(r: dict) -> PositionAnalysis:
    return PositionAnalysis(
        symbol=str(r.get("symbol", "")),
        recommendation=str(r.get("recommendation") or "UNKNOWN"),
        confidence=_number(r.get("confidence")),
        summary=str(r.get("summary") or ""),
        reasons=_strings(r.get("reasons")),
        risks=_strings(r.get("risks")),
        current_price=None if r.get("current_price") is None else _number(r.get("current_price")),
        profit_loss_percent=(None if r.get("profit_loss_percent") is None
                             else _number(r.get("profit_loss_percent"))),
        analysis_type=r.get("analysis_type"),
    )


def _summarize(positions: list[dict]) -> PortfolioSummary | None:
    if not positions:
        return None
    from snaptrade_connector import summarize_portfolio
    return PortfolioSummary(**jsonable(summarize_portfolio(positions)))


def _snaptrade_mode(state: AppState) -> str | None:
    if state.snaptrade is None:
        return None
    return "personal" if state.snaptrade.is_personal else "commercial"


def _personal_key_problem(state: AppState) -> str | None:
    """Why a personal key may not be used here, or None. Commercial keys never have one."""
    if state.snaptrade is None or not state.snaptrade.is_personal:
        return None
    problem = token_problem(state.settings.owner_access_token)
    if problem is None:
        return None
    return (f"This server uses a personal SnapTrade key, which can only ever show its "
            f"owner's account, so connection is disabled until an owner token is set "
            f"({problem}).")


def _snaptrade_state(session: Session, state: AppState) -> SnapTradeState:
    configured = state.snaptrade is not None
    problem = _personal_key_problem(state)
    needs_unlock = (configured and problem is None and state.snaptrade.is_personal
                    and not session.owner_unlocked)
    return SnapTradeState(
        configured=configured,
        mode=_snaptrade_mode(state),
        available=configured and problem is None and not needs_unlock,
        owner_unlock_required=needs_unlock,
        unavailable_reason=problem,
        connection_pending=session.snaptrade_portal_url is not None,
    )


def portfolio_state(session: Session, state: AppState) -> PortfolioState:
    return PortfolioState(
        source=session.portfolio_source,
        positions=[to_position(p) for p in session.positions],
        summary=_summarize(session.positions),
        snaptrade=_snaptrade_state(session, state),
    )


def analysis_state(session: Session) -> AnalysisJobState:
    if session.analysis is None:
        return AnalysisJobState(status="idle")
    snap = session.analysis.snapshot()
    snap["results"] = [to_position_analysis(r) for r in snap["results"]]
    snap.pop("id", None)
    return AnalysisJobState(**snap)


def snaptrade_failure(action: str, error: Exception) -> HTTPException:
    """Log the real error (scrubbed); tell the client only what is safe."""
    logger.warning("SnapTrade %s failed: %s", action, scrub(f"{type(error).__name__}: {error}"))
    status = getattr(error, "status", None)
    if isinstance(status, int):
        return HTTPException(502, f"SnapTrade rejected the request to {action} (HTTP {status}).")
    return HTTPException(502, f"Could not reach SnapTrade to {action}.")


# ------------------------------------------------------------------- routes

router = APIRouter(prefix="/api")


@router.get("/health", response_model=Health)
def health() -> Health:
    return Health(status="ok")


@router.get("/capabilities", response_model=Capabilities)
def capabilities(state: AppState = Depends(app_state)) -> Capabilities:
    analyzer = state.shared_analyzer()
    ai_available = bool(analyzer is not None and getattr(analyzer, "ai_available", False))
    return Capabilities(
        market_data=bool(analyzer is not None and getattr(analyzer, "vector_db", None) is not None),
        ai_backend=analyzer.ai_backend if ai_available else None,
        snaptrade=_snaptrade_mode(state),
    )


@router.post("/stocks/analyze", response_model=StockAnalysis)
def analyze_stock_route(body: AnalyzeStockRequest,
                        state: AppState = Depends(app_state)) -> StockAnalysis:
    from services.stock_analysis import analyze_stock
    return to_stock_analysis(analyze_stock(body.query, state.shared_analyzer()))


@router.get("/portfolio", response_model=PortfolioState)
def get_portfolio(session: Session = Depends(current_session),
                  state: AppState = Depends(app_state)) -> PortfolioState:
    return portfolio_state(session, state)


@router.post("/portfolio/demo", response_model=PortfolioState)
def load_demo(session: Session = Depends(current_session),
              state: AppState = Depends(app_state)) -> PortfolioState:
    from robinhood_portfolio_analyzer import mock_portfolio
    with session.lock:
        session.clear_portfolio()
        session.portfolio_source = "demo"
        session.positions = mock_portfolio()
        return portfolio_state(session, state)


@router.delete("/portfolio", response_model=PortfolioState)
def disconnect(session: Session = Depends(current_session),
               state: AppState = Depends(app_state)) -> PortfolioState:
    with session.lock:
        session.clear_portfolio()
        # Disconnecting also signs the owner out of a personal key.
        session.owner_unlocked = False
        return portfolio_state(session, state)


@router.post("/owner/unlock", status_code=204)
def owner_unlock(body: OwnerUnlockRequest,
                 session: Session = Depends(current_session),
                 state: AppState = Depends(app_state)) -> None:
    """Prove this session is the owner, so it may use a personal SnapTrade key.

    Returns None rather than a Response: FastAPI drops headers set on the
    injected response when a route returns its own Response object, which would
    lose the Set-Cookie for a brand-new session -- the unlock would report
    success and then vanish on the next request.
    """
    problem = _personal_key_problem(state)
    if problem is not None:
        raise HTTPException(409, problem)
    if not verify_owner_token(body.token, state.settings.owner_access_token):
        raise HTTPException(403, "That access token is not valid.")
    with session.lock:
        session.owner_unlocked = True


def _require_snaptrade(state: AppState, session: Session):
    """The connector, if this session may use it; otherwise the reason as an HTTP error."""
    if state.snaptrade is None:
        raise HTTPException(409, "Brokerage connection is not configured on this server.")
    problem = _personal_key_problem(state)
    if problem is not None:
        raise HTTPException(409, problem)
    # A personal key resolves to the owner's account whatever the caller sends,
    # so without this check any visitor could load the owner's holdings.
    if state.snaptrade.is_personal and not session.owner_unlocked:
        raise HTTPException(403, "Enter the owner access token to connect the owner's account.")
    return state.snaptrade


@router.post("/snaptrade/connect", response_model=ConnectResponse)
def snaptrade_connect(session: Session = Depends(current_session),
                      state: AppState = Depends(app_state)) -> ConnectResponse:
    connector = _require_snaptrade(state, session)
    with session.lock:
        try:
            user_id, user_secret, url = connector.start_connection(
                redirect_uri=state.settings.snaptrade_redirect_uri
            )
        except Exception as e:
            raise snaptrade_failure("start a connection", e) from None
        # Held server-side only; the browser receives just the portal URL.
        session.snaptrade_user_id = user_id
        session.snaptrade_user_secret = user_secret
        session.snaptrade_portal_url = url
        return ConnectResponse(portal_url=url)


@router.post("/snaptrade/sync", response_model=PortfolioState)
def snaptrade_sync(session: Session = Depends(current_session),
                   state: AppState = Depends(app_state)) -> PortfolioState:
    connector = _require_snaptrade(state, session)
    with session.lock:
        if session.snaptrade_portal_url is None and session.portfolio_source != "snaptrade":
            raise HTTPException(409, "Start a brokerage connection first.")
        try:
            positions = connector.fetch_positions(
                session.snaptrade_user_id, session.snaptrade_user_secret
            )
        except Exception as e:
            raise snaptrade_failure("load holdings", e) from None
        if not positions:
            raise HTTPException(404, "No stock positions found. Finish connecting your account, then retry.")
        session.portfolio_source = "snaptrade"
        session.positions = positions
        session.snaptrade_portal_url = None  # single-use; the handshake is done
        session.analysis = None
        return portfolio_state(session, state)


@router.post("/portfolio/analysis", response_model=AnalysisJobState, status_code=202)
def start_analysis(session: Session = Depends(current_session),
                   state: AppState = Depends(app_state)) -> AnalysisJobState:
    with session.lock:
        if not session.positions:
            raise HTTPException(409, "Load a portfolio before analyzing it.")
        if session.analysis is not None and session.analysis.active:
            raise HTTPException(409, "An analysis is already running.")
        session.analysis = state.runner.start(session.positions)
        return analysis_state(session)


@router.get("/portfolio/analysis", response_model=AnalysisJobState)
def get_analysis(session: Session = Depends(current_session)) -> AnalysisJobState:
    return analysis_state(session)


@router.get("/portfolio/analysis/report", response_class=PlainTextResponse)
def analysis_report(session: Session = Depends(current_session)) -> PlainTextResponse:
    job = session.analysis
    if job is None or job.status != "done":
        raise HTTPException(409, "No finished analysis to export.")
    from robinhood_portfolio_analyzer import RobinhoodPortfolioAnalyzer
    # The report formats these fields with :.2f / %, and jsonable() turns NaN
    # into None, so default them rather than fail the export.
    results = [
        {**r,
         "confidence": r.get("confidence") or 0,
         "profit_loss_percent": r.get("profit_loss_percent") or 0,
         "summary": r.get("summary") or "",
         "recommendation": r.get("recommendation") or "UNKNOWN"}
        for r in job.snapshot()["results"]
    ]
    report = RobinhoodPortfolioAnalyzer.get_analysis_report(results)
    return PlainTextResponse(
        report,
        headers={"Content-Disposition": 'attachment; filename="portfolio_analysis.txt"'},
    )


# ---------------------------------------------------------------------- app

def _mount_frontend(app: FastAPI, static_dir: Path) -> None:
    """Serve the built React app, falling back to index.html for client routes."""
    index = static_dir / "index.html"
    if not index.is_file():
        logger.warning("no frontend build at %s; serving the API only", static_dir)
        return
    root = static_dir.resolve()
    assets = static_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(404)
        if full_path:
            candidate = (static_dir / full_path).resolve()
            # is_relative_to rejects ../ traversal out of the build directory.
            if candidate.is_relative_to(root) and candidate.is_file():
                return FileResponse(candidate)
        # index.html must not be cached, or a deploy keeps serving old asset hashes.
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def create_app(settings: Settings | None = None,
               analyzer_factory: Callable[[], Any] = default_analyzer_factory,
               snaptrade_factory: Callable[[Settings], Any] = default_snaptrade_factory) -> FastAPI:
    settings = settings or load_settings()
    state = AppState(settings, analyzer_factory, snaptrade_factory)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        state.runner.shutdown()

    app = FastAPI(title="AI Stock Analyzer API", version="1.0.0", lifespan=lifespan)
    app.state.app_state = state
    app.include_router(router)
    _mount_frontend(app, settings.static_dir)
    return app


app = create_app()
