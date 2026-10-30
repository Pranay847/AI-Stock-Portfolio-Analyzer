"""Write the API's OpenAPI schema to a file.

    python -m api.export_openapi web/openapi.json

The React app generates its TypeScript types from this file (npm run gen:api),
and tests/test_api_contract.py fails if it falls out of date with the models.
"""

import json
import sys
from pathlib import Path


def openapi_schema() -> dict:
    from api.main import create_app
    from api.settings import load_settings
    from dataclasses import replace

    # No frontend build is needed to describe the API.
    settings = replace(load_settings(), static_dir=Path("/nonexistent"))
    return create_app(settings, analyzer_factory=lambda: None,
                      snaptrade_factory=lambda s: None).openapi()


def render() -> str:
    return json.dumps(openapi_schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "web/openapi.json")
    out.write_text(render())
    print(f"wrote {out}")
