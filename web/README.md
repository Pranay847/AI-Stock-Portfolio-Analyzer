# Web frontend

React 19 + TypeScript + Vite, served in production by the FastAPI app in `../api`
from the same origin.

## Develop

Run the API and the Vite dev server side by side:

```bash
# terminal 1, repo root
uvicorn api.main:app --reload

# terminal 2, web/
npm install
npm run dev        # http://localhost:5173, proxies /api to :8000
```

The proxy keeps the browser on a single origin, exactly as in production, so the
HttpOnly session cookie behaves the same in both.

## Scripts

| Command | What it does |
|---------|--------------|
| `npm run build` | Typecheck and build to `dist/`, which FastAPI serves |
| `npm test` | Vitest unit tests |
| `npm run typecheck` | TypeScript only |
| `npm run lint` | oxlint |
| `npm run gen:api` | Regenerate `src/api/schema.d.ts` from the FastAPI models |

## API types

`src/api/schema.d.ts` is generated from the Pydantic models in `api/schemas.py`
via `openapi.json` -- never edit it by hand. After changing a model, run
`npm run gen:api`; `tests/test_api_contract.py` fails until you do.

## Charts and colour

Charts are plain HTML so every mark is keyboard-focusable. Gain/loss uses a
blue/red diverging pair rather than green/red, which is indistinguishable under
deuteranopia; every delta also carries a sign and an arrow. Tokens live in
`src/index.css`.
