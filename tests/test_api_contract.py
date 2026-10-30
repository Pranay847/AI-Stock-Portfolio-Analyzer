"""The frontend's TypeScript types are generated from web/openapi.json.

If an API model changes without regenerating them, this fails rather than
letting the React app compile against a stale contract. Fix with:

    cd web && npm run gen:api
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.export_openapi import render

ROOT = Path(__file__).resolve().parent.parent


def test_committed_openapi_matches_the_api_models():
    committed = (ROOT / "web" / "openapi.json").read_text()
    assert committed == render(), "web/openapi.json is stale: run `cd web && npm run gen:api`"


def test_generated_types_exist_for_every_response_model():
    types = (ROOT / "web" / "src" / "api" / "schema.d.ts").read_text()
    for model in ("StockAnalysis", "PortfolioState", "AnalysisJobState", "Capabilities",
                  "ConnectResponse", "PositionAnalysis", "Position", "PortfolioSummary"):
        assert f"{model}: {{" in types, f"{model} missing from schema.d.ts"
