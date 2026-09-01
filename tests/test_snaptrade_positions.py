"""Regression tests for SnapTrade holdings parsing.

These pin the shapes the live API actually returns. /positions/all sends an
object with a `results` array of AccountPosition, whose ticker lives on a
nested instrument and whose numbers arrive as strings - all of which the
first implementation got wrong, silently yielding an empty portfolio.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snaptrade_connector import SnapTradeConnector, summarize_portfolio


class _Resp:
    def __init__(self, body):
        self.body = body


def _connector(rows, mode="personal"):
    """Connector wired to a stub returning the given positions payload."""
    class _Acct:
        def list_user_accounts(self, **kw):
            return _Resp([{"id": "acct-1", "name": "Individual",
                           "institution_name": "Robinhood"}])

        def get_all_account_positions(self, **kw):
            return _Resp(rows)

    class _Client:
        account_information = _Acct()

    c = SnapTradeConnector(client_id="cid", consumer_key="ck", auth_mode=mode)
    c._client = _Client()
    return c


ACCOUNT_POSITION = {
    "results": [
        {
            "instrument": {"kind": "stock", "symbol": "AAPL",
                           "raw_symbol": "AAPL", "description": "Apple Inc."},
            "units": "10", "price": "220.5", "cost_basis": "150.0",
        }
    ],
    "data_freshness": {},
}


def test_positions_all_object_shape():
    """The response is an object with `results`, not a bare list."""
    pos = _connector(ACCOUNT_POSITION).fetch_positions()
    assert len(pos) == 1, f"expected 1 position, got {len(pos)}"
    p = pos[0]
    assert p["symbol"] == "AAPL"
    assert p["name"] == "Apple Inc."
    assert p["quantity"] == 10.0            # string "10" coerced
    assert p["current_price"] == 220.5
    assert p["average_buy_price"] == 150.0  # from cost_basis, per share
    assert round(p["profit_loss_percent"], 2) == 47.0


def test_legacy_holdings_list_shape_still_parses():
    """The older /holdings list-of-Position shape keeps working."""
    legacy = [{
        "symbol": {"symbol": {"symbol": "MSFT", "description": "Microsoft"}},
        "units": 5, "price": 300.0, "average_purchase_price": 250.0,
    }]
    pos = _connector(legacy).fetch_positions()
    assert len(pos) == 1
    assert pos[0]["symbol"] == "MSFT"
    assert pos[0]["average_buy_price"] == 250.0


def test_missing_price_is_not_a_total_loss():
    """A null price must not be reported as -100%."""
    rows = {"results": [{
        "instrument": {"kind": "stock", "symbol": "XYZ", "raw_symbol": "XYZ"},
        "units": "3", "price": None, "cost_basis": "50.0",
    }], "data_freshness": {}}
    p = _connector(rows).fetch_positions()[0]
    assert p["profit_loss_percent"] == 0.0
    assert p["profit_loss"] == 0.0


def test_rows_without_a_symbol_are_skipped():
    rows = {"results": [{"units": "1", "price": "5"}], "data_freshness": {}}
    assert _connector(rows).fetch_positions() == []


def test_summary_matches_analyzer_keys():
    s = summarize_portfolio(_connector(ACCOUNT_POSITION).fetch_positions())
    assert set(s) == {
        "total_positions", "total_equity", "total_cost_basis",
        "total_profit_loss", "total_profit_loss_percent",
        "winners", "losers", "best_performer", "worst_performer",
    }


def test_blank_auth_mode_defaults_to_personal():
    """A blank value must not fall through to commercial (would 403).

    The env-var path is the one that actually regressed: os.getenv returns the
    empty string when the variable is set-but-blank, so the "personal" default
    never applied and the key was used in commercial mode.
    """
    for blank in ("", "   ", None):
        c = SnapTradeConnector(client_id="c", consumer_key="k", auth_mode=blank)
        assert c.is_personal, f"blank auth_mode arg {blank!r} selected {c.auth_mode}"

    prior = os.environ.get("SNAPTRADE_AUTH_MODE")
    try:
        for blank in ("", "   "):
            os.environ["SNAPTRADE_AUTH_MODE"] = blank
            c = SnapTradeConnector(client_id="c", consumer_key="k")
            assert c.is_personal, (
                f"blank SNAPTRADE_AUTH_MODE={blank!r} selected {c.auth_mode!r}"
            )
        # an explicit value is still honoured
        os.environ["SNAPTRADE_AUTH_MODE"] = "commercial"
        assert not SnapTradeConnector(client_id="c", consumer_key="k").is_personal
    finally:
        if prior is None:
            os.environ.pop("SNAPTRADE_AUTH_MODE", None)
        else:
            os.environ["SNAPTRADE_AUTH_MODE"] = prior


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"  FAIL  {name}: {e}")
    print("\nALL PASS" if not failures else f"\n{failures} FAILED")
    sys.exit(1 if failures else 0)
