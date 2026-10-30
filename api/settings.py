"""Runtime configuration, read from the environment (and a local .env)."""

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv is optional in production images
    pass

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    snaptrade_client_id: str | None
    snaptrade_consumer_key: str | None
    snaptrade_auth_mode: str | None
    # Optional. When set, SnapTrade's portal sends the user back here after
    # they connect. Unset keeps the portal open in its own tab instead.
    snaptrade_redirect_uri: str | None
    # Required before a *personal* SnapTrade key can be used: that key always
    # resolves to its owner's account, so visitors must not reach it. See
    # services/owner_access.py.
    owner_access_token: str | None
    static_dir: Path
    session_ttl_seconds: int
    max_sessions: int
    max_concurrent_analyses: int

    @property
    def snaptrade_configured(self) -> bool:
        return bool(self.snaptrade_client_id and self.snaptrade_consumer_key)


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def load_settings() -> Settings:
    return Settings(
        snaptrade_client_id=os.getenv("SNAPTRADE_CLIENT_ID") or None,
        snaptrade_consumer_key=os.getenv("SNAPTRADE_CONSUMER_KEY") or None,
        snaptrade_auth_mode=os.getenv("SNAPTRADE_AUTH_MODE") or None,
        snaptrade_redirect_uri=os.getenv("SNAPTRADE_REDIRECT_URI") or None,
        owner_access_token=os.getenv("OWNER_ACCESS_TOKEN") or None,
        static_dir=Path(os.getenv("STATIC_DIR", REPO_ROOT / "web" / "dist")),
        session_ttl_seconds=_int("SESSION_TTL_SECONDS", 2 * 60 * 60),
        max_sessions=_int("MAX_SESSIONS", 1000),
        max_concurrent_analyses=_int("MAX_CONCURRENT_ANALYSES", 2),
    )
