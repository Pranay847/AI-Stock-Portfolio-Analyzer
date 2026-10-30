import type {
  AnalysisJobState,
  Capabilities,
  ConnectResponse,
  PortfolioState,
  StockAnalysis,
} from './types'

/** A non-2xx response, carrying the server's message when it sent one. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

type ValidationIssue = { msg?: string; loc?: unknown[] }

/** FastAPI sends `detail` as a string, or as a list of issues for a 422. */
function messageFrom(detail: unknown, status: number): string {
  if (typeof detail === 'string' && detail) return detail
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((issue: ValidationIssue) => issue?.msg)
      .filter(Boolean)
      .join('; ')
  }
  return `Request failed (HTTP ${status})`
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api${path}`, {
      // Same origin in every environment (Vite proxies /api in development),
      // so the HttpOnly session cookie is sent without any CORS setup.
      credentials: 'same-origin',
      ...init,
      headers: {
        Accept: 'application/json',
        ...(init.body ? { 'Content-Type': 'application/json' } : {}),
        ...init.headers,
      },
    })
  } catch {
    throw new ApiError(0, 'Could not reach the server. Check your connection and try again.')
  }

  if (!response.ok) {
    let detail: unknown
    try {
      detail = ((await response.json()) as { detail?: unknown }).detail
    } catch {
      // Not JSON -- fall back to a generic message.
    }
    throw new ApiError(response.status, messageFrom(detail, response.status))
  }
  // 204 No Content has no body to parse.
  if (response.status === 204) return null as T
  return (await response.json()) as T
}

export const api = {
  capabilities: () => request<Capabilities>('/capabilities'),

  analyzeStock: (query: string) =>
    request<StockAnalysis>('/stocks/analyze', {
      method: 'POST',
      body: JSON.stringify({ query }),
    }),

  portfolio: () => request<PortfolioState>('/portfolio'),
  loadDemo: () => request<PortfolioState>('/portfolio/demo', { method: 'POST' }),
  disconnect: () => request<PortfolioState>('/portfolio', { method: 'DELETE' }),

  /** Prove this session is the owner, so it may use a personal SnapTrade key. */
  unlockOwner: (token: string) =>
    request<null>('/owner/unlock', { method: 'POST', body: JSON.stringify({ token }) }),

  connectSnapTrade: () => request<ConnectResponse>('/snaptrade/connect', { method: 'POST' }),
  syncSnapTrade: () => request<PortfolioState>('/snaptrade/sync', { method: 'POST' }),

  startAnalysis: () => request<AnalysisJobState>('/portfolio/analysis', { method: 'POST' }),
  analysis: () => request<AnalysisJobState>('/portfolio/analysis'),
}

/** A plain link target: the browser downloads it via Content-Disposition. */
export const REPORT_URL = '/api/portfolio/analysis/report'
