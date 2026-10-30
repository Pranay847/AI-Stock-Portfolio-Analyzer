import {
  AlertCircle,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Info,
  Loader2,
  Minus,
} from 'lucide-react'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

import { direction, type Direction } from '../lib/format'

function cx(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ')
}

// ---------------------------------------------------------------- layout

export function Card({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: ReactNode
  description?: ReactNode
  actions?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cx('rounded-xl border border-border bg-surface p-5', className)}>
      {(title || actions) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            {title && <h2 className="text-base font-semibold text-text">{title}</h2>}
            {description && <p className="mt-1 text-sm text-muted">{description}</p>}
          </div>
          {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

// --------------------------------------------------------------- buttons

type Variant = 'primary' | 'secondary' | 'ghost'

const variants: Record<Variant, string> = {
  primary: 'bg-accent text-accent-fg hover:brightness-110',
  secondary: 'border border-border bg-surface-2 text-text hover:border-axis',
  ghost: 'text-muted hover:bg-surface-2 hover:text-text',
}

export function Button({
  variant = 'primary',
  loading = false,
  icon,
  children,
  className,
  disabled,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant
  loading?: boolean
  icon?: ReactNode
}) {
  return (
    <button
      type="button"
      {...rest}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cx(
        'inline-flex h-10 items-center justify-center gap-2 rounded-lg px-4 text-sm font-semibold',
        'transition focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent-text',
        'disabled:cursor-not-allowed disabled:opacity-55',
        variants[variant],
        className,
      )}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden /> : icon}
      {children}
    </button>
  )
}

export function Spinner({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-sm text-muted" role="status">
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </span>
  )
}

// --------------------------------------------------------------- callouts

type Tone = 'info' | 'warn' | 'error'

const tones: Record<Tone, { box: string; icon: ReactNode }> = {
  info: { box: 'border-border bg-surface-2', icon: <Info className="size-4 text-accent-text" aria-hidden /> },
  warn: { box: 'border-warn/40 bg-warn/10', icon: <AlertTriangle className="size-4 text-warn" aria-hidden /> },
  error: { box: 'border-loss/40 bg-loss/10', icon: <AlertCircle className="size-4 text-loss-text" aria-hidden /> },
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: Tone
  title: ReactNode
  children?: ReactNode
}) {
  return (
    <div
      role={tone === 'error' ? 'alert' : 'status'}
      className={cx('flex gap-3 rounded-lg border p-4 text-sm', tones[tone].box)}
    >
      <span className="mt-0.5 shrink-0">{tones[tone].icon}</span>
      <div className="min-w-0">
        <p className="font-semibold text-text">{title}</p>
        {children && <div className="mt-1 text-muted">{children}</div>}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- figures

const directionText: Record<Direction, string> = {
  gain: 'text-gain-text',
  loss: 'text-loss-text',
  flat: 'text-muted',
}

/** A signed change. Colour never carries it alone: there is always a sign and an arrow. */
export function Delta({ value, children }: { value: number | null | undefined; children: ReactNode }) {
  const dir = direction(value)
  const Icon = dir === 'gain' ? ArrowUpRight : dir === 'loss' ? ArrowDownRight : Minus
  return (
    <span className={cx('inline-flex items-center gap-0.5 font-medium', directionText[dir])}>
      <Icon className="size-3.5 shrink-0" aria-hidden />
      {children}
    </span>
  )
}

/** label · value · optional delta -- the stat-tile contract. */
export function StatTile({
  label,
  value,
  delta,
  hint,
}: {
  label: string
  value: ReactNode
  delta?: ReactNode
  hint?: ReactNode
}) {
  return (
    <div className="rounded-lg border border-border bg-surface-2 px-4 py-3">
      <p className="text-xs text-muted">{label}</p>
      {/* Proportional figures: tabular-nums makes large numbers look loose. */}
      <p className="mt-1 text-xl font-semibold text-text">{value}</p>
      {delta && <p className="mt-0.5 text-sm">{delta}</p>}
      {hint && <p className="mt-0.5 text-xs text-muted">{hint}</p>}
    </div>
  )
}

/** A ratio against a limit, on a lighter step of the fill's own hue. */
export function Meter({
  value,
  max = 100,
  label,
  valueText,
  size = 'md',
}: {
  value: number
  max?: number
  label: string
  valueText?: string
  size?: 'sm' | 'md'
}) {
  const clamped = Math.max(0, Math.min(max, Number.isFinite(value) ? value : 0))
  const pct = max > 0 ? (clamped / max) * 100 : 0
  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={clamped}
      aria-valuetext={valueText}
      className={cx(
        'w-full overflow-hidden rounded-full',
        '[background:color-mix(in_oklab,var(--gain)_22%,var(--surface))]',
        size === 'sm' ? 'h-1.5' : 'h-2',
      )}
    >
      <div className="h-full rounded-full bg-gain" style={{ width: `${pct}%` }} />
    </div>
  )
}

// ---------------------------------------------------------------- signals

const signals = {
  BUY: { icon: ArrowUpRight, box: 'border-gain/40 bg-gain/12 text-gain-text' },
  SELL: { icon: ArrowDownRight, box: 'border-loss/40 bg-loss/12 text-loss-text' },
  HOLD: { icon: Minus, box: 'border-border bg-surface-2 text-text' },
} as const

/** BUY / SELL / HOLD, with an icon so the call never depends on colour. */
export function SignalBadge({ signal, large = false }: { signal: string; large?: boolean }) {
  const key = (signal.toUpperCase() in signals ? signal.toUpperCase() : 'HOLD') as keyof typeof signals
  const { icon: Icon, box } = signals[key]
  const known = signal.toUpperCase() in signals
  return (
    <span
      className={cx(
        'inline-flex items-center gap-1 rounded-md border font-semibold tracking-wide',
        large ? 'px-3 py-1.5 text-lg' : 'px-2 py-0.5 text-xs',
        known ? box : 'border-border bg-surface-2 text-muted',
      )}
    >
      <Icon className={large ? 'size-5' : 'size-3.5'} aria-hidden />
      {known ? key : signal.toUpperCase()}
    </span>
  )
}

export function EmptyState({ icon, title, children }: { icon: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 px-6 py-10 text-center">
      <span className="text-muted" aria-hidden>
        {icon}
      </span>
      <p className="font-semibold text-text">{title}</p>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
    </div>
  )
}
