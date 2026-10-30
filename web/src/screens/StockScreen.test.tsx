import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { StockAnalysis } from '../api/types'
import { StockScreen } from './StockScreen'

const base: StockAnalysis = {
  ok: true,
  symbol: 'AAPL',
  current_price: 200,
  buy_price: 194,
  sell_price: 226,
  recommendation: 'BUY',
  confidence: 72,
  holding_period: 'Long Term (6+ months)',
  summary: 'Strong momentum.',
  reasons: ['Momentum', 'Momentum'],
  risks: ['Valuation'],
  xgboost_signal: 'BUY',
  xgboost_confidence: 0.71,
  analysis_type: 'langgraph_workflow',
  ai_model: 'LangChain + Mistral',
  as_of: '2026-10-28',
  market_state: 'CLOSED',
  quote_source: 'yahoo',
}

function renderWith(result: Partial<StockAnalysis>) {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ ...base, ...result }), { status: 200 })))
  const client = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <StockScreen />
    </QueryClientProvider>,
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('StockScreen', () => {
  it('renders a successful analysis', async () => {
    renderWith({})
    await userEvent.type(screen.getByLabelText('Ticker or company name'), 'apple')
    await userEvent.click(screen.getByRole('button', { name: 'Analyze' }))

    expect(await screen.findByText('$200.00')).toBeInTheDocument()
    expect(screen.getAllByText('BUY').length).toBeGreaterThan(0)
    expect(screen.getByText('Quote as of 2026-10-28 · closed · via yahoo')).toBeInTheDocument()
    expect(screen.getByRole('meter', { name: 'Recommendation confidence' })).toHaveAttribute('aria-valuenow', '72')
    // Duplicate LLM lines still render (keys include the index).
    expect(screen.getAllByText('Momentum')).toHaveLength(2)
  })

  it('renders a failed analysis as a failure, with no price or signal', async () => {
    renderWith({
      ok: false,
      current_price: 0,
      recommendation: 'HOLD',
      summary: 'Market data is not configured, so no quote could be requested.',
      reasons: ['ALPHA_VANTAGE_API_KEY is not set for this deployment'],
    })
    await userEvent.click(screen.getByRole('button', { name: 'AAPL' }))

    expect(await screen.findByText('Market data is not configured, so no quote could be requested.')).toBeInTheDocument()
    expect(screen.getByText('ALPHA_VANTAGE_API_KEY is not set for this deployment')).toBeInTheDocument()
    expect(screen.queryByText('$0.00')).toBeNull()
    expect(screen.queryByText('HOLD')).toBeNull()
  })

  it('does not submit an empty or whitespace query', async () => {
    renderWith({})
    const button = screen.getByRole('button', { name: 'Analyze' })
    expect(button).toBeDisabled()
    await userEvent.type(screen.getByLabelText('Ticker or company name'), '   ')
    expect(button).toBeDisabled()
    expect(fetch).not.toHaveBeenCalled()
  })
})
