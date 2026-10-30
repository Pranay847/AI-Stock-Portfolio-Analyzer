const MINUS = '−' // a true minus sign, the width of "+", so signed columns align

const usd = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

const usdCompact = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  notation: 'compact',
  maximumFractionDigits: 1,
})

const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 })

/** $1,234.56 -- or $12.9K / $4.2M from 10,000 up, for stat tiles. */
export function money(value: number, { compact = false } = {}): string {
  if (!Number.isFinite(value)) return '—'
  if (compact && Math.abs(value) >= 10_000) return usdCompact.format(value)
  return usd.format(value)
}

/** +$250.00 / −$25.00 */
export function signedMoney(value: number, opts?: { compact?: boolean }): string {
  if (!Number.isFinite(value)) return '—'
  const sign = value > 0 ? '+' : value < 0 ? MINUS : ''
  return sign + money(Math.abs(value), opts)
}

/** 16.67% -- no sign. */
export function percent(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—'
  return `${value.toFixed(digits)}%`
}

/** +16.67% / −3.57% */
export function signedPercent(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '—'
  const sign = value > 0 ? '+' : value < 0 ? MINUS : ''
  return `${sign}${Math.abs(value).toFixed(digits)}%`
}

export function quantity(value: number): string {
  return Number.isFinite(value) ? number.format(value) : '—'
}

export type Direction = 'gain' | 'loss' | 'flat'

export function direction(value: number | null | undefined): Direction {
  if (value == null || !Number.isFinite(value) || value === 0) return 'flat'
  return value > 0 ? 'gain' : 'loss'
}

/** Percentage change from `from` to `to`; null when undefined (from = 0). */
export function change(from: number, to: number): number | null {
  if (!from) return null
  return ((to - from) / from) * 100
}
