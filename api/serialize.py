"""Make analysis output safe to send as JSON.

Results come from pandas, numpy and xgboost, so they can carry numpy scalars
and NaN/inf. Starlette serialises responses with allow_nan=False, so a single
NaN in a P/L figure would otherwise turn a successful analysis into a 500.
"""

import math
from datetime import date, datetime
from typing import Any


def jsonable(value: Any) -> Any:
    """Recursively convert to JSON-native types; non-finite floats become None."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    # numpy scalars and 0-d arrays expose .item(); arrays expose .tolist().
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)
