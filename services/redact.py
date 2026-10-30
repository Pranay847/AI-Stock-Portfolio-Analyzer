"""Strip credentials from text bound for logs or the screen."""

import re

# userSecret travels as a query parameter on commercial SnapTrade calls, and the
# SDK's connection errors quote the request URL.
_SECRET_PARAM = re.compile(r"(userSecret=)[^&\s'\"]+", re.IGNORECASE)


def scrub(text: str) -> str:
    return _SECRET_PARAM.sub(r"\1[redacted]", text)
