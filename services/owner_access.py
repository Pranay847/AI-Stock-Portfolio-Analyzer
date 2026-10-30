"""Who may use a personal SnapTrade key.

A personal key identifies its owner by itself: SnapTrade resolves every call to
the owner's brokerage account, with no per-visitor identity. So on a public
deployment, letting any visitor "connect" with it would show the owner's real
holdings to whoever clicked. Connection is therefore gated on an owner access
token: unset means the feature is off; set means a session must present it.

Commercial keys need no gate -- each visitor registers their own SnapTrade user
and sees only the account they connect.
"""

import hmac

# Long enough that guessing it online is hopeless; anything shorter is refused
# rather than silently accepted as protection.
MIN_TOKEN_LENGTH = 16


def token_problem(token: str | None) -> str | None:
    """Why `token` cannot protect a personal key, or None if it can."""
    if not token:
        return "OWNER_ACCESS_TOKEN is not set"
    if len(token) < MIN_TOKEN_LENGTH:
        return f"OWNER_ACCESS_TOKEN must be at least {MIN_TOKEN_LENGTH} characters"
    return None


def verify_owner_token(supplied: str | None, expected: str | None) -> bool:
    """Constant-time comparison; always False when no usable token is configured."""
    if token_problem(expected) is not None or not supplied:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
