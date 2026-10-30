"""Request and response models: the contract the React app is typed against.

web/src/api/types.ts mirrors these; keep the two in step.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Capabilities(BaseModel):
    market_data: bool = Field(description="A quote provider is configured")
    ai_backend: str | None = Field(description="Active LLM backend, or null for rule-based only")
    snaptrade: Literal["personal", "commercial"] | None = Field(
        description="SnapTrade key type, or null if brokerage connection is not configured"
    )


class Health(BaseModel):
    """Liveness only, so container health checks never wait on analyzer start-up."""
    status: Literal["ok"]


class AnalyzeStockRequest(BaseModel):
    query: str = Field(min_length=1, max_length=64, description="Ticker or company name")


class StockAnalysis(BaseModel):
    ok: bool = Field(description="False when no analysis could be produced; see summary/reasons")
    symbol: str
    current_price: float
    buy_price: float
    sell_price: float
    recommendation: str
    confidence: float
    holding_period: str
    summary: str
    reasons: list[str] = []
    risks: list[str] = []
    xgboost_signal: str | None = None
    xgboost_confidence: float | None = None
    analysis_type: str
    ai_model: str
    as_of: str | None = None
    market_state: str | None = None
    quote_source: str | None = None


class Position(BaseModel):
    symbol: str
    name: str
    quantity: float
    average_buy_price: float
    current_price: float
    equity: float
    cost_basis: float
    profit_loss: float
    profit_loss_percent: float


class PortfolioSummary(BaseModel):
    total_positions: int
    total_equity: float
    total_cost_basis: float
    total_profit_loss: float
    total_profit_loss_percent: float
    winners: int
    losers: int
    best_performer: str | None
    worst_performer: str | None


class SnapTradeState(BaseModel):
    configured: bool
    mode: Literal["personal", "commercial"] | None
    # This session may start a connection now.
    available: bool
    # Personal key: the session must present the owner access token first.
    owner_unlock_required: bool
    # Why connection is off despite being configured (e.g. no owner token).
    unavailable_reason: str | None
    # True between starting a connection and loading holdings.
    connection_pending: bool


class OwnerUnlockRequest(BaseModel):
    token: str = Field(min_length=1, max_length=256)


class PortfolioState(BaseModel):
    source: Literal["demo", "snaptrade"] | None
    positions: list[Position]
    summary: PortfolioSummary | None
    snaptrade: SnapTradeState


class ConnectResponse(BaseModel):
    portal_url: str


class PositionAnalysis(BaseModel):
    symbol: str
    recommendation: str
    confidence: float
    summary: str
    reasons: list[str] = []
    risks: list[str] = []
    current_price: float | None = None
    profit_loss_percent: float | None = None
    analysis_type: str | None = None


class AnalysisJobState(BaseModel):
    status: Literal["idle", "queued", "running", "done", "error"]
    total: int = 0
    completed: int = 0
    current_symbol: str | None = None
    results: list[PositionAnalysis] = []
    notes: list[str] = []
    error: str | None = None
