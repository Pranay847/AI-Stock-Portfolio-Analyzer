"""Server-side sessions.

The browser holds only an opaque random session id in an HttpOnly cookie. All
state -- loaded holdings, and above all the SnapTrade userSecret -- stays on the
server. (Starlette's SessionMiddleware is deliberately not used: it stores data
in a signed but *readable* cookie, which would put the userSecret in the
browser.)

State lives in process memory, so it is lost on restart and is not shared
between replicas. That is fine for a single-container deployment; scaling out
needs a shared store such as Redis behind the same interface.
"""

import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

COOKIE_NAME = "sid"
_ID_BYTES = 32


@dataclass
class Session:
    id: str
    touched: float = field(default_factory=time.monotonic)
    # "demo" or "snaptrade" once holdings are loaded.
    portfolio_source: str | None = None
    positions: list[dict] = field(default_factory=list)
    # SnapTrade handshake. user_id/user_secret are only used with commercial
    # keys; a personal key identifies its owner by itself.
    snaptrade_user_id: str | None = None
    snaptrade_user_secret: str | None = None
    snaptrade_portal_url: str | None = None
    # Set once this session presents the owner access token; required before a
    # personal SnapTrade key may be used (services/owner_access.py).
    owner_unlocked: bool = False
    # The current or most recent portfolio analysis (api.jobs.AnalysisJob).
    analysis: Any = None
    # Serialises mutations from concurrent requests in the same session
    # (a double-clicked button, two open tabs).
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def clear_portfolio(self) -> None:
        self.portfolio_source = None
        self.positions = []
        self.snaptrade_user_id = None
        self.snaptrade_user_secret = None
        self.snaptrade_portal_url = None
        self.analysis = None


class SessionStore:
    def __init__(self, ttl_seconds: int, max_sessions: int):
        self._ttl = ttl_seconds
        self._max = max_sessions
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        """Return (session, created).

        An id the store does not hold -- expired, or never issued -- gets a
        brand-new random id rather than being adopted, so a client cannot pick
        its own session id (session fixation).
        """
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            session = self._sessions.get(session_id) if session_id else None
            if session is not None:
                session.touched = now
                self._sessions.move_to_end(session.id)
                return session, False

            session = Session(id=secrets.token_urlsafe(_ID_BYTES), touched=now)
            self._sessions[session.id] = session
            while len(self._sessions) > self._max:
                self._sessions.popitem(last=False)  # least recently used
            return session, True

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _evict_expired(self, now: float) -> None:
        # OrderedDict is kept in last-touched order, so stop at the first live one.
        while self._sessions:
            oldest = next(iter(self._sessions.values()))
            if now - oldest.touched <= self._ttl:
                break
            self._sessions.popitem(last=False)
