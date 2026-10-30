import { describe, expect, it } from 'vitest'

import { change, direction, money, percent, signedMoney, signedPercent } from './format'

describe('format', () => {
  it('formats money, compacting only from 10,000 when asked', () => {
    expect(money(1234.5)).toBe('$1,234.50')
    expect(money(12_900, { compact: true })).toBe('$12.9K')
    expect(money(9_999, { compact: true })).toBe('$9,999.00')
    expect(money(4_200_000, { compact: true })).toBe('$4.2M')
  })

  it('signs with a true minus sign so columns align', () => {
    expect(signedPercent(-3.57)).toBe('−3.57%')
    expect(signedPercent(16.67)).toBe('+16.67%')
    expect(signedPercent(0)).toBe('0.00%')
    expect(signedMoney(-25)).toBe('−$25.00')
    expect(signedMoney(250)).toBe('+$250.00')
  })

  it('renders non-finite values as a dash, never NaN', () => {
    expect(money(Number.NaN)).toBe('—')
    expect(percent(Number.POSITIVE_INFINITY)).toBe('—')
    expect(signedPercent(Number.NaN)).toBe('—')
  })

  it('classifies direction', () => {
    expect(direction(2)).toBe('gain')
    expect(direction(-2)).toBe('loss')
    expect(direction(0)).toBe('flat')
    expect(direction(null)).toBe('flat')
    expect(direction(Number.NaN)).toBe('flat')
  })

  it('computes change, refusing to divide by zero', () => {
    expect(change(100, 110)).toBeCloseTo(10)
    expect(change(0, 110)).toBeNull()
  })
})
