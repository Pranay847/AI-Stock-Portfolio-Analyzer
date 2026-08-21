"""
Secure brokerage connection via SnapTrade.

Lets a user connect their Robinhood (or other brokerage) account without this
app ever seeing their credentials. The user authenticates inside SnapTrade's
hosted connection portal; this app only ever holds a per-user access pair
(userId + userSecret) that grants read access to positions.

Configuration (both required, e.g. in Streamlit secrets or .env):
    SNAPTRADE_CLIENT_ID
    SNAPTRADE_CONSUMER_KEY

Positions are normalised to the same shape used by
robinhood_portfolio_analyzer.fetch_portfolio(), so the rest of the analysis
pipeline works unchanged.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

try:
    from snaptrade_client import SnapTrade, SnapTradeAuth
    SNAPTRADE_SDK_AVAILABLE = True
except ImportError:
    SnapTrade = None
    SnapTradeAuth = None
    SNAPTRADE_SDK_AVAILABLE = False


def _body(response):
    """Return the parsed body of an SDK response."""
    return getattr(response, "body", response)


def _get(obj, *keys, default=None):
    """Safely walk nested dict/attr paths, returning the first hit."""
    for key in keys:
        current = obj
        for part in key.split("."):
            if current is None:
                break
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        if current is not None:
            return current
    return default


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class SnapTradeConnector:
    """Thin wrapper over the SnapTrade SDK for read-only portfolio access."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        consumer_key: Optional[str] = None,
        auth_mode: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("SNAPTRADE_CLIENT_ID")
        self.consumer_key = consumer_key or os.getenv("SNAPTRADE_CONSUMER_KEY")
        # "personal" (free key, your own accounts only) or "commercial" (paid,
        # lets any end user connect their own account). Defaults to personal so
        # the free key works out of the box.
        # See https://docs.snaptrade.com/docs/personal-vs-commercial
        self.auth_mode = (
            auth_mode or os.getenv("SNAPTRADE_AUTH_MODE", "personal")
        ).strip().lower()
        self._client = None

    @property
    def is_configured(self) -> bool:
        """True when the SDK is installed and API keys are present."""
        return bool(SNAPTRADE_SDK_AVAILABLE and self.client_id and self.consumer_key)

    @property
    def client(self):
        if self._client is None:
            if not self.is_configured:
                raise RuntimeError(
                    "SnapTrade is not configured. Set SNAPTRADE_CLIENT_ID and "
                    "SNAPTRADE_CONSUMER_KEY, and install snaptrade-python-sdk."
                )
            make_auth = (
                SnapTradeAuth.personal_api_key
                if self.auth_mode.startswith("personal")
                else SnapTradeAuth.commercial_api_key
            )
            self._client = SnapTrade(
                auth=make_auth(
                    consumer_key=self.consumer_key,
                    client_id=self.client_id,
                )
            )
        return self._client

    @property
    def is_personal(self) -> bool:
        """True for a personal key, which represents the key owner directly."""
        return self.auth_mode.startswith("personal")

    def _user_kwargs(self, user_id: Optional[str], user_secret: Optional[str]) -> dict:
        """Per-user credentials, omitted entirely for personal keys.

        A personal key resolves the user from the key itself and rejects
        userId/userSecret, so they must not be sent.
        """
        if self.is_personal:
            return {}
        return {"user_id": user_id, "user_secret": user_secret}

    # ---------------- user registration ----------------

    def register_user(self, user_id: Optional[str] = None) -> tuple[str, str]:
        """Register an end user with SnapTrade (commercial keys only).

        Returns:
            (user_id, user_secret) - the secret is the credential for later
            reads and should be held only for the length of the session.
        """
        if self.is_personal:
            raise RuntimeError(
                "Personal API keys represent you directly and cannot register "
                "users. Use start_connection() instead."
            )
        user_id = user_id or f"user-{uuid.uuid4()}"
        resp = self.client.authentication.register_snap_trade_user(user_id=user_id)
        body = _body(resp)
        user_secret = _get(body, "userSecret", "user_secret")
        if not user_secret:
            raise RuntimeError("SnapTrade did not return a userSecret")
        return user_id, str(user_secret)

    def start_connection(
        self, redirect_uri: Optional[str] = None, broker: str = "ROBINHOOD"
    ) -> tuple[Optional[str], Optional[str], str]:
        """Begin a brokerage connection.

        Returns:
            (user_id, user_secret, connection_url). For a personal key the two
            ids are None, since the key itself identifies the user.
        """
        if self.is_personal:
            user_id = user_secret = None
        else:
            user_id, user_secret = self.register_user()
        url = self.get_connection_url(
            user_id, user_secret, redirect_uri=redirect_uri, broker=broker
        )
        return user_id, user_secret, url

    # ---------------- connection portal ----------------

    def get_connection_url(
        self,
        user_id: Optional[str] = None,
        user_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        broker: str = "ROBINHOOD",
    ) -> str:
        """Create a one-time URL for SnapTrade's hosted connection portal.

        The user logs into their brokerage on that page, not in this app.
        """
        kwargs = self._user_kwargs(user_id, user_secret)
        if broker:
            kwargs["broker"] = broker
        if redirect_uri:
            kwargs["custom_redirect"] = redirect_uri

        resp = self.client.authentication.login_snap_trade_user(**kwargs)
        body = _body(resp)
        url = _get(body, "redirectURI", "redirect_uri", "redirectUri")
        if not url:
            raise RuntimeError("SnapTrade did not return a connection URL")
        return str(url)

    # ---------------- reading the portfolio ----------------

    def list_accounts(
        self, user_id: Optional[str] = None, user_secret: Optional[str] = None
    ) -> list[dict]:
        """List the brokerage accounts the user has connected."""
        resp = self.client.account_information.list_user_accounts(
            **self._user_kwargs(user_id, user_secret)
        )
        accounts = _body(resp) or []
        return [
            {
                "id": str(_get(a, "id", default="")),
                "name": str(_get(a, "name", "number", default="Account")),
                "institution": str(_get(a, "institution_name", "institutionName", default="")),
            }
            for a in accounts
        ]

    def fetch_positions(
        self,
        user_id: Optional[str] = None,
        user_secret: Optional[str] = None,
        account_id: Optional[str] = None,
    ) -> list[dict]:
        """Fetch holdings, normalised to the app's position shape.

        When account_id is None, positions from every connected account are
        combined.
        """
        if account_id:
            account_ids = [account_id]
        else:
            account_ids = [a["id"] for a in self.list_accounts(user_id, user_secret)]

        portfolio = []
        for acct_id in account_ids:
            if not acct_id:
                continue
            resp = self.client.account_information.get_all_account_positions(
                account_id=acct_id, **self._user_kwargs(user_id, user_secret)
            )
            for position in (_body(resp) or []):
                normalised = self._normalise_position(position)
                if normalised:
                    portfolio.append(normalised)
        return portfolio

    @staticmethod
    def _normalise_position(position) -> Optional[dict]:
        """Map a SnapTrade position onto the app's position dict."""
        symbol = _get(
            position,
            "symbol.symbol.symbol",     # universal symbol, the usual shape
            "symbol.symbol",
            "symbol.raw_symbol",
            "symbol",
        )
        if not symbol or not isinstance(symbol, str):
            return None

        name = _get(
            position,
            "symbol.symbol.description",
            "symbol.description",
            default=symbol,
        )

        quantity = _to_float(_get(position, "units", "quantity"))
        current_price = _to_float(_get(position, "price", "last_price"))
        avg_buy_price = _to_float(
            _get(position, "average_purchase_price", "averagePurchasePrice")
        )

        equity = quantity * current_price
        cost_basis = quantity * avg_buy_price
        profit_loss = equity - cost_basis
        profit_loss_pct = (
            ((current_price - avg_buy_price) / avg_buy_price * 100)
            if avg_buy_price > 0
            else 0.0
        )

        return {
            "symbol": symbol,
            "name": str(name),
            "quantity": quantity,
            "average_buy_price": avg_buy_price,
            "current_price": current_price,
            "equity": equity,
            "cost_basis": cost_basis,
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_pct,
            "timestamp": datetime.now().isoformat(),
        }


def summarize_portfolio(portfolio: list[dict]) -> dict:
    """Summary statistics matching RobinhoodPortfolioAnalyzer.get_portfolio_summary()."""
    if not portfolio:
        return {}

    total_equity = sum(p["equity"] for p in portfolio)
    total_cost = sum(p["cost_basis"] for p in portfolio)
    total_pl = sum(p["profit_loss"] for p in portfolio)
    total_pl_pct = ((total_equity - total_cost) / total_cost * 100) if total_cost > 0 else 0

    return {
        "total_positions": len(portfolio),
        "total_equity": total_equity,
        "total_cost_basis": total_cost,
        "total_profit_loss": total_pl,
        "total_profit_loss_percent": total_pl_pct,
        "winners": len([p for p in portfolio if p["profit_loss"] > 0]),
        "losers": len([p for p in portfolio if p["profit_loss"] < 0]),
        "best_performer": max(portfolio, key=lambda x: x["profit_loss_percent"])["symbol"],
        "worst_performer": min(portfolio, key=lambda x: x["profit_loss_percent"])["symbol"],
    }
