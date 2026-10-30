import { useQuery } from '@tanstack/react-query'
import type { MouseEvent, ReactNode } from 'react'

import { api } from './api/client'
import { Callout } from './components/ui'
import { useRoute, type Route } from './lib/route'
import { PortfolioScreen } from './screens/PortfolioScreen'
import { StockScreen } from './screens/StockScreen'

const NAV: Array<{ route: Route; href: string; label: string }> = [
  { route: 'stock', href: '/', label: 'Stock analysis' },
  { route: 'portfolio', href: '/portfolio', label: 'My portfolio' },
]

export default function App() {
  const [route, navigate] = useRoute()
  const capabilities = useQuery({ queryKey: ['capabilities'], queryFn: api.capabilities, staleTime: 5 * 60_000 })
  const caps = capabilities.data

  function go(event: MouseEvent<HTMLAnchorElement>, next: Route) {
    // Real links, so middle-click and copy-link work; plain clicks stay in the app.
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) return
    event.preventDefault()
    navigate(next)
  }

  return (
    <div className="min-h-screen">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-8 gap-y-3 px-4 py-3 sm:px-6">
          <a href="/" onClick={(e) => go(e, 'stock')} className="flex items-center gap-2 text-text">
            <img src="/favicon.svg" alt="" className="size-7" />
            <span className="font-semibold">AI Stock Analyzer</span>
          </a>
          <nav aria-label="Main" className="flex gap-1">
            {NAV.map((item) => {
              const current = route === item.route
              return (
                <a
                  key={item.route}
                  href={item.href}
                  onClick={(e) => go(e, item.route)}
                  aria-current={current ? 'page' : undefined}
                  className={[
                    'rounded-md px-3 py-1.5 text-sm font-medium',
                    current ? 'bg-surface-2 text-text' : 'text-muted hover:text-text',
                  ].join(' ')}
                >
                  {item.label}
                </a>
              )
            })}
          </nav>
          {caps && (
            <dl className="ml-auto flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
              <Capability label="Market data" value={caps.market_data ? 'Connected' : 'Not configured'} ok={caps.market_data} />
              <Capability label="AI" value={caps.ai_backend ?? 'Rule-based only'} ok={caps.ai_backend !== null} />
            </dl>
          )}
        </div>
      </header>

      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        {caps && !caps.market_data && route === 'stock' && (
          <Callout tone="warn" title="Market data is not configured on this deployment">
            Stock analysis needs a quote provider key (ALPHA_VANTAGE_API_KEY), so lookups will report that no quote is
            available. The portfolio demo still works.
          </Callout>
        )}
        {route === 'stock' ? <StockScreen /> : <PortfolioScreen />}
      </main>

      <footer className="mx-auto max-w-6xl px-4 pb-10 text-xs text-muted sm:px-6">
        XGBoost signals with retrieval-augmented LLM reasoning. For research and learning, not financial advice.
      </footer>
    </div>
  )
}

function Capability({ label, value, ok }: { label: string; value: ReactNode; ok: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      {/* A shape as well as a colour: filled when on, hollow when off. */}
      <span
        aria-hidden
        className={['size-2 rounded-full', ok ? 'bg-gain' : 'border border-muted'].join(' ')}
      />
      <dt>{label}:</dt>
      <dd className="m-0 text-text">{value}</dd>
    </div>
  )
}
