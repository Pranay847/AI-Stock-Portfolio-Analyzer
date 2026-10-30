import { useMutation } from '@tanstack/react-query'
import { Brain, ChartNoAxesColumn, Cpu, Search } from 'lucide-react'
import { useState, type FormEvent } from 'react'

import { api } from '../api/client'
import type { StockAnalysis } from '../api/types'
import { Button, Callout, Card, Delta, Meter, SignalBadge, Spinner, StatTile } from '../components/ui'
import { change, money, percent, signedPercent } from '../lib/format'

const QUICK_PICKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA']

const PIPELINE = [
  {
    icon: <ChartNoAxesColumn className="size-4" aria-hidden />,
    title: 'Technical features',
    body: 'Moving averages, returns, volatility and trend signals from years of price history.',
  },
  {
    icon: <Cpu className="size-4" aria-hidden />,
    title: 'XGBoost signal',
    body: 'A gradient-boosted classifier scores the ticker BUY, SELL or HOLD with a confidence.',
  },
  {
    icon: <Brain className="size-4" aria-hidden />,
    title: 'Plain-English reasoning',
    body: 'Retrieved market context goes to an LLM, so each call comes with an explanation.',
  },
]

export function StockScreen() {
  const [query, setQuery] = useState('')
  // useMutation clears `data` as soon as a new mutation starts, so the last
  // result is held here to stay on screen (dimmed) while the next one runs.
  const [last, setLast] = useState<StockAnalysis | null>(null)
  const analysis = useMutation({
    mutationFn: (q: string) => api.analyzeStock(q),
    onSuccess: setLast,
  })

  function submit(q: string) {
    const trimmed = q.trim()
    if (!trimmed || analysis.isPending) return
    setQuery(trimmed)
    analysis.mutate(trimmed)
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    submit(query)
  }

  return (
    <div className="space-y-6">
      <Card title="Analyze a stock" description="Enter a ticker or a company name.">
        <form onSubmit={onSubmit} className="flex flex-col gap-3 sm:flex-row">
          <label className="sr-only" htmlFor="stock-query">
            Ticker or company name
          </label>
          <input
            id="stock-query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="AAPL, Apple, Microsoft…"
            maxLength={64}
            autoComplete="off"
            spellCheck={false}
            className="h-10 min-w-0 flex-1 rounded-lg border border-border bg-surface-2 px-3 text-sm text-text placeholder:text-muted focus:border-accent-text focus:outline-none"
          />
          <Button type="submit" loading={analysis.isPending} disabled={!query.trim()} icon={<Search className="size-4" aria-hidden />}>
            Analyze
          </Button>
        </form>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-muted">
          <span>Try:</span>
          {QUICK_PICKS.map((symbol) => (
            <button
              key={symbol}
              type="button"
              onClick={() => submit(symbol)}
              disabled={analysis.isPending}
              className="rounded-md border border-border px-2 py-1 font-medium text-text hover:border-axis disabled:opacity-55"
            >
              {symbol}
            </button>
          ))}
        </div>
      </Card>

      {analysis.isPending && (
        <Spinner label={`Analyzing ${query.toUpperCase()}… an AI analysis can take up to a minute.`} />
      )}

      {analysis.isError && (
        <Callout tone="error" title="The analysis could not be run">
          {analysis.error.message}
        </Callout>
      )}

      {last && !analysis.isError && (
        <div
          aria-busy={analysis.isPending || undefined}
          className={analysis.isPending ? 'opacity-50 transition-opacity' : 'transition-opacity'}
        >
          <AnalysisResult result={last} />
        </div>
      )}

      {analysis.isIdle && (
        <ul className="m-0 grid list-none gap-3 p-0 sm:grid-cols-3">
          {PIPELINE.map((step) => (
            <li key={step.title} className="rounded-xl border border-border bg-surface p-4">
              <p className="flex items-center gap-2 text-sm font-semibold text-text">
                <span className="text-accent-text">{step.icon}</span>
                {step.title}
              </p>
              <p className="mt-1.5 text-sm text-muted">{step.body}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function AnalysisResult({ result }: { result: StockAnalysis }) {
  if (!result.ok) {
    return (
      <Callout tone="warn" title={result.summary || `No analysis for ${result.symbol}`}>
        {result.reasons.length > 0 && (
          <ul className="mt-1 list-disc space-y-0.5 pl-5">
            {result.reasons.map((reason, i) => (
              <li key={`${i}-${reason}`}>{reason}</li>
            ))}
          </ul>
        )}
      </Callout>
    )
  }

  const price = result.current_price
  const buyChange = change(price, result.buy_price)
  const sellChange = change(price, result.sell_price)
  const xgbConfidence =
    result.xgboost_confidence == null
      ? null
      : result.xgboost_confidence <= 1
        ? result.xgboost_confidence * 100
        : result.xgboost_confidence
  const quoteBits = [
    result.as_of && `Quote as of ${result.as_of}`,
    result.market_state && result.market_state.toLowerCase(),
    result.quote_source && `via ${result.quote_source}`,
  ].filter(Boolean)

  return (
    <Card>
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-muted">{result.symbol}</p>
          {/* The view's one hero figure. */}
          <p className="text-5xl font-semibold tracking-tight text-text">{money(price)}</p>
          {quoteBits.length > 0 && <p className="mt-1 text-xs text-muted">{quoteBits.join(' · ')}</p>}
        </div>
        <div className="text-right">
          <SignalBadge signal={result.recommendation} large />
          <p className="mt-1 text-xs text-muted">Analyzed by {result.ai_model}</p>
        </div>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile
          label="Buy target"
          value={money(result.buy_price)}
          delta={buyChange != null && <Delta value={buyChange}>{signedPercent(buyChange, 1)} vs current</Delta>}
        />
        <StatTile
          label="Sell target"
          value={money(result.sell_price)}
          delta={sellChange != null && <Delta value={sellChange}>{signedPercent(sellChange, 1)} vs current</Delta>}
        />
        <StatTile label="Holding period" value={<span className="text-base">{result.holding_period}</span>} />
        <StatTile
          label="XGBoost signal"
          value={result.xgboost_signal ? <SignalBadge signal={result.xgboost_signal} /> : <span className="text-base text-muted">No model</span>}
          hint={xgbConfidence != null ? `${percent(xgbConfidence, 0)} model confidence` : 'No trained model is deployed'}
        />
      </div>

      <div className="mt-5">
        <div className="mb-1.5 flex items-baseline justify-between text-sm">
          <span className="text-muted">Recommendation confidence</span>
          <span className="font-semibold text-text">{percent(result.confidence, 0)}</span>
        </div>
        <Meter value={result.confidence} label="Recommendation confidence" valueText={percent(result.confidence, 0)} />
      </div>

      {result.summary && <p className="mt-5 text-sm leading-relaxed text-text">{result.summary}</p>}

      {(result.reasons.length > 0 || result.risks.length > 0) && (
        <div className="mt-5 grid gap-5 sm:grid-cols-2">
          <ReasonList title="Why" items={result.reasons} />
          <ReasonList title="Risks" items={result.risks} />
        </div>
      )}

      <p className="mt-6 border-t border-border pt-3 text-xs text-muted">
        Model output for research and learning, not financial advice.
      </p>
    </Card>
  )
}

function ReasonList({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null
  return (
    <div>
      <h3 className="text-xs font-semibold tracking-wide text-muted uppercase">{title}</h3>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-text">
        {items.map((item, i) => (
          // Index in the key: an LLM can repeat a line verbatim.
          <li key={`${i}-${item}`}>{item}</li>
        ))}
      </ul>
    </div>
  )
}
