import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { signedPercent } from '../lib/format'
import { DivergingBars } from './DivergingBars'

const items = [
  { key: 'TSLA', label: 'TSLA', value: 25, details: [{ label: 'Gain / loss', value: '+$400.00' }] },
  { key: 'GOOGL', label: 'GOOGL', value: -3.57, details: [{ label: 'Gain / loss', value: '−$25.00' }] },
]

describe('DivergingBars', () => {
  it('labels every bar with its value, so nothing depends on hovering', () => {
    render(<DivergingBars caption="Return" items={items} format={(v) => signedPercent(v)} />)
    expect(screen.getByText('+25.00%')).toBeInTheDocument()
    expect(screen.getByText('−3.57%')).toBeInTheDocument()
  })

  it('makes each row focusable with an accessible name', () => {
    render(<DivergingBars caption="Return" items={items} format={(v) => signedPercent(v)} />)
    const rows = screen.getAllByRole('listitem')
    expect(rows).toHaveLength(2)
    expect(rows[0]).toHaveAttribute('tabindex', '0')
    expect(rows[0]).toHaveAccessibleName('TSLA: +25.00%')
  })

  it('shows the same tooltip on keyboard focus as on hover', () => {
    render(<DivergingBars caption="Return" items={items} format={(v) => signedPercent(v)} />)
    const [tsla, googl] = screen.getAllByRole('listitem')
    expect(screen.queryByRole('tooltip')).toBeNull()

    fireEvent.focus(tsla)
    expect(screen.getByRole('tooltip')).toHaveTextContent('+$400.00')
    expect(tsla).toHaveAttribute('aria-describedby', screen.getByRole('tooltip').id)

    fireEvent.blur(tsla)
    fireEvent.pointerEnter(googl)
    expect(screen.getByRole('tooltip')).toHaveTextContent('−$25.00')
    fireEvent.pointerLeave(googl)
    expect(screen.queryByRole('tooltip')).toBeNull()
  })

  it('places zero by the data range, so an all-gain chart has no empty left arm', () => {
    const { container } = render(
      <DivergingBars caption="Return" items={[{ key: 'A', label: 'A', value: 10 }]} format={String} />,
    )
    const baseline = container.querySelector('.bg-axis') as HTMLElement
    expect(baseline.style.left).toBe('0%')
  })
})
