import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Download, ExternalLink, KeyRound, Link2, LogOut, Play, RefreshCw, Sparkles, Wallet } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'

import { api, REPORT_URL } from '../api/client'
import type { AnalysisJobState, PortfolioState, Position, SnapTradeState } from '../api/types'
import { DivergingBars } from '../components/DivergingBars'
import { Button, Callout, Card, Delta, EmptyState, Meter, SignalBadge, Spinner, StatTile } from '../components/ui'
import { money, percent, quantity, signedMoney, signedPercent } from '../lib/format'

const PORTFOLIO_KEY = ['portfolio'] as const
const ANALYSIS_KEY = ['analysis'] as const

export function PortfolioScreen() {
  const queryClient = useQueryClient()
  const portfolio = useQuery({ queryKey: PORTFOLIO_KEY, queryFn: api.portfolio })

  const setPortfolio = (state: PortfolioState) => {
    queryClient.setQueryData(PORTFOLIO_KEY, state)
    // Loading or clearing holdings resets any analysis on the server.
    void queryClient.invalidateQueries({ queryKey: ANALYSIS_KEY })
  }

  if (portfolio.isPending) return <Spinner label="Loading your portfolio…" />
  if (portfolio.isError) {
    return (
      <Callout tone="error" title="Could not load your portfolio">
        {portfolio.error.message}
      </Callout>
    )
  }

  const state = portfolio.data
  if (!state.source) return <ConnectPanel state={state} onLoaded={setPortfolio} />
  return <PortfolioView state={state} onCleared={setPortfolio} />
}

// ------------------------------------------------------------- connecting

function ConnectPanel({ state, onLoaded }: { state: PortfolioState; onLoaded: (s: PortfolioState) => void }) {
  const demo = useMutation({ mutationFn: api.loadDemo, onSuccess: onLoaded })

  return (
    <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
      <Card
        title="Connect your brokerage"
        description="You sign in on SnapTrade's secure page. This app never sees your Robinhood password."
      >
        <SnapTradeConnect snaptrade={state.snaptrade} onLoaded={onLoaded} />
      </Card>

      <Card title="Or explore a demo" description="Four sample positions, no account needed.">
        <Button variant="secondary" loading={demo.isPending} onClick={() => demo.mutate()} icon={<Wallet className="size-4" aria-hidden />}>
          Load demo portfolio
        </Button>
        {demo.isError && (
          <div className="mt-3">
            <Callout tone="error" title="Could not load the demo">
              {demo.error.message}
            </Callout>
          </div>
        )}
      </Card>
    </div>
  )
}

function SnapTradeConnect({ snaptrade, onLoaded }: { snaptrade: SnapTradeState; onLoaded: (s: PortfolioState) => void }) {
  const queryClient = useQueryClient()
  const [portalUrl, setPortalUrl] = useState<string | null>(null)
  const refreshPortfolio = () => queryClient.invalidateQueries({ queryKey: PORTFOLIO_KEY })

  const connect = useMutation({
    mutationFn: api.connectSnapTrade,
    onSuccess: (res) => {
      setPortalUrl(res.portal_url)
      void refreshPortfolio()
    },
  })
  const sync = useMutation({ mutationFn: api.syncSnapTrade, onSuccess: onLoaded })
  const cancel = useMutation({
    mutationFn: api.disconnect,
    onSuccess: (s) => {
      setPortalUrl(null)
      onLoaded(s)
    },
  })

  // With SNAPTRADE_REDIRECT_URI pointing at /portfolio?snaptrade=connected,
  // the portal sends the user back here; load their holdings once on arrival.
  const autoSynced = useRef(false)
  useEffect(() => {
    const params = new URLSearchParams(window.location.search)
    if (params.get('snaptrade') !== 'connected' || autoSynced.current) return
    autoSynced.current = true
    window.history.replaceState(null, '', window.location.pathname)
    if (snaptrade.available) sync.mutate()
  }, [snaptrade.available, sync])

  if (!snaptrade.configured) {
    return <p className="text-sm text-muted">Brokerage connection is not set up on this deployment. Try the demo instead.</p>
  }
  if (snaptrade.unavailable_reason) {
    return (
      <div className="space-y-2 text-sm">
        <p className="text-muted">Brokerage connection is not enabled on this deployment. Try the demo instead.</p>
        <details className="text-xs text-muted">
          <summary className="cursor-pointer">Why?</summary>
          <p className="mt-1">{snaptrade.unavailable_reason}</p>
        </details>
      </div>
    )
  }
  if (snaptrade.owner_unlock_required) return <OwnerUnlock />

  const pending = portalUrl !== null || snaptrade.connection_pending
  if (!pending) {
    return (
      <div className="space-y-3">
        <Button loading={connect.isPending} onClick={() => connect.mutate()} icon={<Link2 className="size-4" aria-hidden />}>
          Connect Robinhood
        </Button>
        {connect.isError && (
          <Callout tone="error" title="Could not start the connection">
            {connect.error.message}
          </Callout>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <ol className="m-0 list-decimal space-y-1 pl-5 text-sm text-text">
        <li>Open the secure login and connect your Robinhood account.</li>
        <li>Come back here and load your holdings.</li>
      </ol>
      <div className="flex flex-wrap gap-2">
        {portalUrl ? (
          <a
            href={portalUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-10 items-center gap-2 rounded-lg bg-accent px-4 text-sm font-semibold text-accent-fg hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-text"
          >
            <ExternalLink className="size-4" aria-hidden />
            Open secure login
          </a>
        ) : (
          <Button variant="secondary" loading={connect.isPending} onClick={() => connect.mutate()} icon={<RefreshCw className="size-4" aria-hidden />}>
            Get a new login link
          </Button>
        )}
        <Button variant="secondary" loading={sync.isPending} onClick={() => sync.mutate()}>
          I've connected — load my holdings
        </Button>
        <Button variant="ghost" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          Cancel
        </Button>
      </div>
      {sync.isError && (
        <Callout tone={sync.error.message.includes('No stock positions') ? 'warn' : 'error'} title="Could not load your holdings">
          {sync.error.message}
        </Callout>
      )}
    </div>
  )
}

function OwnerUnlock() {
  const queryClient = useQueryClient()
  const [token, setToken] = useState('')
  const unlock = useMutation({
    mutationFn: (t: string) => api.unlockOwner(t),
    onSuccess: () => {
      setToken('')
      void queryClient.invalidateQueries({ queryKey: PORTFOLIO_KEY })
    },
  })

  function onSubmit(event: FormEvent) {
    event.preventDefault()
    if (token) unlock.mutate(token)
  }

  return (
    <form onSubmit={onSubmit} className="space-y-3">
      <p className="text-sm text-muted">
        This deployment uses the owner's personal brokerage key, so only the owner can connect.
      </p>
      <label htmlFor="owner-token" className="block text-sm font-medium text-text">
        Owner access token
      </label>
      <div className="flex flex-col gap-2 sm:flex-row">
        <input
          id="owner-token"
          type="password"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          autoComplete="off"
          className="h-10 min-w-0 flex-1 rounded-lg border border-border bg-surface-2 px-3 text-sm text-text focus:border-accent-text focus:outline-none"
        />
        <Button type="submit" loading={unlock.isPending} disabled={!token} icon={<KeyRound className="size-4" aria-hidden />}>
          Unlock
        </Button>
      </div>
      {unlock.isError && (
        <Callout tone="error" title="Could not unlock">
          {unlock.error.message}
        </Callout>
      )}
    </form>
  )
}

// -------------------------------------------------------------- portfolio

function PortfolioView({ state, onCleared }: { state: PortfolioState; onCleared: (s: PortfolioState) => void }) {
  const disconnect = useMutation({ mutationFn: api.disconnect, onSuccess: onCleared })
  const summary = state.summary
  const positions = [...state.positions].sort((a, b) => b.equity - a.equity)
  const totalEquity = summary?.total_equity ?? 0

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-sm text-muted">{state.source === 'demo' ? 'Demo portfolio' : 'Connected via SnapTrade'}</p>
          {/* The view's one hero figure. */}
          <p className="text-5xl font-semibold tracking-tight text-text">{money(totalEquity)}</p>
          {summary && (
            <p className="mt-1 text-sm">
              <Delta value={summary.total_profit_loss}>
                {signedMoney(summary.total_profit_loss)} ({signedPercent(summary.total_profit_loss_percent)}) all time
              </Delta>
            </p>
          )}
        </div>
        <Button variant="secondary" loading={disconnect.isPending} onClick={() => disconnect.mutate()} icon={<LogOut className="size-4" aria-hidden />}>
          {state.source === 'demo' ? 'Close demo' : 'Disconnect'}
        </Button>
      </div>

      {summary && (
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <StatTile label="Cost basis" value={money(summary.total_cost_basis, { compact: true })} />
          <StatTile label="Positions" value={summary.total_positions} hint={`${summary.winners} up · ${summary.losers} down`} />
          <StatTile label="Best performer" value={summary.best_performer ?? '—'} />
          <StatTile label="Worst performer" value={summary.worst_performer ?? '—'} />
        </div>
      )}

      <Card>
        <DivergingBars
          caption="Return by position"
          description="Unrealized gain or loss against average cost. Hover or focus a row for details; the table below lists every value."
          format={(v) => signedPercent(v)}
          items={[...positions]
            .sort((a, b) => b.profit_loss_percent - a.profit_loss_percent)
            .map((p) => ({
              key: p.symbol,
              label: p.symbol,
              value: p.profit_loss_percent,
              details: [
                { label: 'Gain / loss', value: signedMoney(p.profit_loss) },
                { label: 'Market value', value: money(p.equity) },
                { label: 'Weight', value: totalEquity ? percent((p.equity / totalEquity) * 100, 1) : '—' },
              ],
            }))}
        />
      </Card>

      <HoldingsTable positions={positions} totalEquity={totalEquity} />
      <AnalysisPanel />
    </div>
  )
}

function HoldingsTable({ positions, totalEquity }: { positions: Position[]; totalEquity: number }) {
  return (
    <Card title="Holdings">
      <div className="-mx-5 overflow-x-auto">
        <table className="tabular w-full min-w-[720px] border-collapse text-sm">
          <thead>
            <tr className="border-b border-border text-left text-xs text-muted">
              <th scope="col" className="px-5 py-2 font-medium">Symbol</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Shares</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Avg cost</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Price</th>
              <th scope="col" className="px-3 py-2 text-right font-medium">Value</th>
              <th scope="col" className="w-36 px-3 py-2 font-medium">Weight</th>
              <th scope="col" className="px-5 py-2 text-right font-medium">Gain / loss</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((p) => {
              const weight = totalEquity ? (p.equity / totalEquity) * 100 : 0
              return (
                <tr key={p.symbol} className="border-b border-border last:border-0">
                  <th scope="row" className="px-5 py-2.5 text-left font-normal">
                    <span className="font-semibold text-text">{p.symbol}</span>
                    {p.name && p.name !== p.symbol && <span className="block text-xs text-muted">{p.name}</span>}
                  </th>
                  <td className="px-3 py-2.5 text-right text-text">{quantity(p.quantity)}</td>
                  <td className="px-3 py-2.5 text-right text-text">{money(p.average_buy_price)}</td>
                  <td className="px-3 py-2.5 text-right text-text">{money(p.current_price)}</td>
                  <td className="px-3 py-2.5 text-right font-medium text-text">{money(p.equity)}</td>
                  <td className="px-3 py-2.5">
                    <div className="flex items-center gap-2">
                      <Meter value={weight} label={`${p.symbol} weight`} valueText={percent(weight, 1)} size="sm" />
                      <span className="w-12 shrink-0 text-right text-xs text-muted">{percent(weight, 1)}</span>
                    </div>
                  </td>
                  <td className="px-5 py-2.5 text-right">
                    <Delta value={p.profit_loss}>
                      {signedMoney(p.profit_loss)} ({signedPercent(p.profit_loss_percent)})
                    </Delta>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

// --------------------------------------------------------------- analysis

const isActive = (s?: AnalysisJobState) => s?.status === 'queued' || s?.status === 'running'

function AnalysisPanel() {
  const queryClient = useQueryClient()
  const analysis = useQuery({
    queryKey: ANALYSIS_KEY,
    queryFn: api.analysis,
    // Poll only while a job is in flight.
    refetchInterval: (query) => (isActive(query.state.data) ? 1000 : false),
  })
  const start = useMutation({
    mutationFn: api.startAnalysis,
    onSuccess: (job) => queryClient.setQueryData(ANALYSIS_KEY, job),
  })

  const job = analysis.data
  const running = isActive(job)

  return (
    <Card
      title="AI analysis"
      description="Runs every position through the analysis pipeline. With an LLM configured this can take a few minutes."
      actions={
        <>
          {job?.status === 'done' && job.results.length > 0 && (
            <a
              href={REPORT_URL}
              download
              className="inline-flex h-10 items-center gap-2 rounded-lg border border-border bg-surface-2 px-4 text-sm font-semibold text-text hover:border-axis"
            >
              <Download className="size-4" aria-hidden />
              Export report
            </a>
          )}
          <Button
            loading={start.isPending}
            disabled={running}
            onClick={() => start.mutate()}
            icon={job?.status === 'done' ? <RefreshCw className="size-4" aria-hidden /> : <Play className="size-4" aria-hidden />}
          >
            {job?.status === 'done' ? 'Run again' : 'Analyze portfolio'}
          </Button>
        </>
      }
    >
      {start.isError && (
        <div className="mb-4">
          <Callout tone="error" title="Could not start the analysis">
            {start.error.message}
          </Callout>
        </div>
      )}

      {running && job && (
        <div className="space-y-2" aria-live="polite">
          <div className="flex justify-between text-sm">
            <span className="text-muted">
              {job.status === 'queued' ? 'Waiting for a free worker…' : `Analyzing${job.current_symbol ? ` ${job.current_symbol}` : ''}…`}
            </span>
            <span className="font-medium text-text">
              {job.completed} / {job.total}
            </span>
          </div>
          <Meter value={job.completed} max={Math.max(job.total, 1)} label="Analysis progress" valueText={`${job.completed} of ${job.total} positions`} />
        </div>
      )}

      {job?.status === 'error' && (
        <Callout tone="error" title="The analysis failed">
          {job.error}
        </Callout>
      )}

      {job?.status === 'done' && (
        <div className="space-y-4">
          {job.notes.length > 0 && (
            <Callout tone="warn" title="Some positions were not analyzed">
              <ul className="mt-1 list-disc pl-5">
                {job.notes.map((note, i) => (
                  <li key={`${i}-${note}`}>{note}</li>
                ))}
              </ul>
            </Callout>
          )}
          <ul className="m-0 list-none divide-y divide-border p-0">
            {job.results.map((r) => (
              <li key={r.symbol} className="grid gap-2 py-3 sm:grid-cols-[6rem_8rem_1fr] sm:items-start">
                <span className="font-semibold text-text">{r.symbol}</span>
                <div className="space-y-1.5">
                  <SignalBadge signal={r.recommendation} />
                  <Meter value={r.confidence} label={`${r.symbol} confidence`} valueText={percent(r.confidence, 0)} size="sm" />
                  <span className="block text-xs text-muted">{percent(r.confidence, 0)} confidence</span>
                </div>
                <p className="m-0 text-sm text-text">{r.summary}</p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {(!job || job.status === 'idle') && !running && (
        <EmptyState icon={<Sparkles className="size-6" />} title="No analysis yet">
          Get a BUY, SELL or HOLD call with reasoning for each position.
        </EmptyState>
      )}
    </Card>
  )
}
