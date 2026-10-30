import { useId, useState, type ReactNode } from 'react'

export type DivergingItem = {
  key: string
  label: string
  value: number
  /** Extra tooltip rows, e.g. the dollar P/L and equity behind the percentage. */
  details?: Array<{ label: string; value: string }>
}

/**
 * Horizontal diverging bars around zero -- for "above or below a baseline".
 *
 * Plain HTML rather than a chart library so every row is a real focusable
 * element: hover and keyboard focus show the same tooltip. Every value is also
 * printed at the bar tip, and the holdings table is the full table-view twin, so
 * the tooltip only ever adds detail.
 */
export function DivergingBars({
  items,
  format,
  caption,
  description,
}: {
  items: DivergingItem[]
  format: (value: number) => string
  caption: ReactNode
  description?: ReactNode
}) {
  const [active, setActive] = useState<string | null>(null)
  const tooltipId = useId()

  const values = items.map((i) => (Number.isFinite(i.value) ? i.value : 0))
  const lo = Math.min(0, ...values)
  const hi = Math.max(0, ...values)
  const span = hi - lo || 1
  // Zero sits where the data puts it: an all-gain portfolio gets no empty left arm.
  const zero = (-lo / span) * 100

  return (
    <figure className="m-0">
      <figcaption className="mb-3">
        <span className="text-sm font-semibold text-text">{caption}</span>
        {description && <span className="mt-0.5 block text-xs text-muted">{description}</span>}
      </figcaption>

      <ul className="m-0 list-none space-y-0.5 p-0">
        {items.map((item) => {
          const value = Number.isFinite(item.value) ? item.value : 0
          const width = (Math.abs(value) / span) * 100
          const gain = value >= 0
          const isActive = active === item.key
          return (
            <li
              key={item.key}
              tabIndex={0}
              aria-label={`${item.label}: ${format(value)}`}
              aria-describedby={isActive ? tooltipId : undefined}
              onPointerEnter={() => setActive(item.key)}
              onPointerLeave={() => setActive((k) => (k === item.key ? null : k))}
              onFocus={() => setActive(item.key)}
              onBlur={() => setActive((k) => (k === item.key ? null : k))}
              className="relative grid cursor-default grid-cols-[4.5rem_1fr] items-center rounded-md py-1.5 outline-none focus-visible:ring-2 focus-visible:ring-accent-text"
            >
              <span className="truncate pr-2 text-sm font-medium text-text">{item.label}</span>

              {/* px-16 reserves room either side so tip labels are never clipped. */}
              <div className="relative h-5 px-16">
                <div className="relative h-full">
                  <div className="absolute inset-y-0 w-px bg-axis" style={{ left: `${zero}%` }} aria-hidden />
                  {width > 0 && (
                    <div
                      aria-hidden
                      className={[
                        'absolute top-1/2 h-3.5 -translate-y-1/2 transition-[filter]',
                        // 4px rounded data-end, square at the baseline.
                        gain ? 'rounded-r bg-gain' : 'rounded-l bg-loss',
                        isActive ? 'brightness-125' : '',
                      ].join(' ')}
                      style={gain ? { left: `${zero}%`, width: `${width}%` } : { right: `${100 - zero}%`, width: `${width}%` }}
                    />
                  )}
                  {/* Value at the tip, in text ink -- never the bar colour. */}
                  <span
                    className="absolute top-1/2 -translate-y-1/2 whitespace-nowrap text-xs font-medium text-text"
                    style={
                      gain
                        ? { left: `calc(${zero + width}% + 0.375rem)` }
                        : { right: `calc(${100 - zero + width}% + 0.375rem)` }
                    }
                  >
                    {format(value)}
                  </span>
                </div>
              </div>

              {isActive && item.details && item.details.length > 0 && (
                <div
                  id={tooltipId}
                  role="tooltip"
                  className="pointer-events-none absolute bottom-full left-[4.5rem] z-10 mb-1 min-w-44 rounded-lg border border-border bg-surface-2 px-3 py-2 shadow-lg"
                >
                  {/* Values lead, labels follow. */}
                  <p className="text-sm font-semibold text-text">{format(value)}</p>
                  <p className="text-xs text-muted">{item.label}</p>
                  <dl className="mt-1.5 space-y-0.5 text-xs">
                    {item.details.map((d) => (
                      <div key={d.label} className="flex justify-between gap-4">
                        <dt className="text-muted">{d.label}</dt>
                        <dd className="m-0 font-medium text-text">{d.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              )}
            </li>
          )
        })}
      </ul>
    </figure>
  )
}
