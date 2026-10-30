import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { PortfolioState, SnapTradeState } from '../api/types'
import { PortfolioScreen } from './PortfolioScreen'

type Handler = () => Response | Promise<Response>

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })

/** Route fetch by "METHOD /path"; anything unrouted fails the test loudly. */
function route(handlers: Record<string, Handler>) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? 'GET'} ${url}`
      const handler = handlers[key]
      if (!handler) throw new Error(`unrouted request: ${key}`)
      return handler()
    }),
  )
}

const snaptrade = (over: Partial<SnapTradeState> = {}): SnapTradeState => ({
  configured: true,
  mode: 'commercial',
  available: true,
  owner_unlock_required: false,
  unavailable_reason: null,
  connection_pending: false,
  ...over,
})

const empty = (over: Partial<SnapTradeState> = {}): PortfolioState => ({
  source: null,
  positions: [],
  summary: null,
  snaptrade: snaptrade(over),
})

const demo: PortfolioState = {
  source: 'demo',
  positions: [
    { symbol: 'AAPL', name: 'Apple Inc', quantity: 10, average_buy_price: 150, current_price: 175,
      equity: 1750, cost_basis: 1500, profit_loss: 250, profit_loss_percent: 16.67 },
    // The same symbol held in a second account must not collide on React keys.
    { symbol: 'AAPL', name: 'Apple Inc', quantity: 1, average_buy_price: 180, current_price: 175,
      equity: 175, cost_basis: 180, profit_loss: -5, profit_loss_percent: -2.78 },
  ],
  summary: { total_positions: 2, total_equity: 1925, total_cost_basis: 1680, total_profit_loss: 245,
    total_profit_loss_percent: 14.58, winners: 1, losers: 1, best_performer: 'AAPL', worst_performer: 'AAPL' },
  snaptrade: snaptrade(),
}

function renderScreen() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <PortfolioScreen />
    </QueryClientProvider>,
  )
  return client
}

afterEach(() => vi.unstubAllGlobals())

describe('PortfolioScreen connect flow', () => {
  it('shows why "Get a new login link" failed while a connection is pending', async () => {
    route({
      'GET /api/portfolio': () => json(200, empty({ connection_pending: true })),
      'POST /api/snaptrade/connect': () => json(502, { detail: 'Could not reach SnapTrade to start a connection.' }),
    })
    renderScreen()
    await userEvent.click(await screen.findByRole('button', { name: 'Get a new login link' }))
    expect(await screen.findByText('Could not reach SnapTrade to start a connection.')).toBeInTheDocument()
  })

  it('offers a fresh login link while one is already held', async () => {
    let n = 0
    route({
      'GET /api/portfolio': () => json(200, empty({ connection_pending: n > 0 })),
      'POST /api/snaptrade/connect': () => json(200, { portal_url: `https://portal.example/${++n}` }),
    })
    renderScreen()
    await userEvent.click(await screen.findByRole('button', { name: 'Connect Robinhood' }))
    const link = await screen.findByRole('link', { name: 'Open secure login' })
    expect(link).toHaveAttribute('href', 'https://portal.example/1')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')

    await userEvent.click(screen.getByRole('button', { name: 'New link' }))
    await waitFor(() =>
      expect(screen.getByRole('link', { name: 'Open secure login' })).toHaveAttribute('href', 'https://portal.example/2'),
    )
  })

  it('shows a failed load even when no connection is pending', async () => {
    route({
      'GET /api/portfolio': () => json(200, empty({ connection_pending: true })),
      'POST /api/snaptrade/sync': () => json(409, { detail: 'Start a brokerage connection first.' }),
    })
    renderScreen()
    await userEvent.click(await screen.findByRole('button', { name: /load my holdings/ }))
    expect(await screen.findByText('Start a brokerage connection first.')).toBeInTheDocument()
  })
})

describe('PortfolioScreen resilience', () => {
  it('keeps the last good data when a background refresh fails', async () => {
    let fail = false
    route({
      'GET /api/portfolio': () => (fail ? json(500, { detail: 'boom' }) : json(200, empty())),
    })
    const client = renderScreen()
    expect(await screen.findByRole('heading', { name: 'Connect your brokerage' })).toBeInTheDocument()

    fail = true
    await client.refetchQueries({ queryKey: ['portfolio'] })
    expect(await screen.findByText('Could not refresh your portfolio')).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Connect your brokerage' })).toBeInTheDocument()
  })

  it('reports a failed disconnect instead of swallowing it', async () => {
    route({
      'GET /api/portfolio': () => json(200, demo),
      'GET /api/portfolio/analysis': () => json(200, { status: 'idle', total: 0, completed: 0, results: [], notes: [] }),
      'DELETE /api/portfolio': () => json(500, { detail: 'Server unavailable' }),
    })
    renderScreen()
    await userEvent.click(await screen.findByRole('button', { name: 'Close demo' }))
    expect(await screen.findByText('Could not disconnect')).toBeInTheDocument()
  })

  it('renders the same symbol held twice without dropping a row', async () => {
    route({
      'GET /api/portfolio': () => json(200, demo),
      'GET /api/portfolio/analysis': () => json(200, { status: 'idle', total: 0, completed: 0, results: [], notes: [] }),
    })
    const errors = vi.spyOn(console, 'error').mockImplementation(() => {})
    renderScreen()
    await screen.findByText('Demo portfolio')
    expect(screen.getAllByRole('row')).toHaveLength(3) // header + 2 positions
    expect(errors.mock.calls.some((c) => String(c[0]).includes('same key'))).toBe(false)
    errors.mockRestore()
  })
})
