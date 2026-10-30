import { afterEach, describe, expect, it, vi } from 'vitest'

import { api, ApiError } from './client'

function respond(status: number, body?: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () =>
      new Response(body === undefined ? null : JSON.stringify(body), {
        status,
        headers: { 'Content-Type': 'application/json' },
      }),
    ),
  )
}

afterEach(() => vi.unstubAllGlobals())

describe('api client', () => {
  it('sends JSON to same-origin /api with credentials', async () => {
    respond(200, { ok: true })
    await api.analyzeStock('AAPL')
    const [url, init] = vi.mocked(fetch).mock.calls[0]
    expect(url).toBe('/api/stocks/analyze')
    expect(init).toBeDefined()
    const sent = init as RequestInit
    expect(sent.method).toBe('POST')
    expect(sent.credentials).toBe('same-origin')
    expect(sent.body).toBe(JSON.stringify({ query: 'AAPL' }))
    expect((sent.headers as Record<string, string>)['Content-Type']).toBe('application/json')
  })

  it("surfaces the server's message on errors", async () => {
    respond(409, { detail: 'Load a portfolio before analyzing it.' })
    await expect(api.startAnalysis()).rejects.toMatchObject({
      status: 409,
      message: 'Load a portfolio before analyzing it.',
    })
  })

  it('flattens FastAPI validation errors (a list) into one message', async () => {
    respond(422, { detail: [{ msg: 'String should have at least 1 character' }, { msg: 'second' }] })
    await expect(api.analyzeStock('')).rejects.toThrow('String should have at least 1 character; second')
  })

  it('falls back to a generic message when the body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('<html>bad gateway</html>', { status: 502 })))
    await expect(api.portfolio()).rejects.toThrow('Request failed (HTTP 502)')
  })

  it('reports an unreachable server distinctly', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Promise.reject(new TypeError('Failed to fetch'))))
    const error = await api.portfolio().catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(0)
  })

  it('treats 204 No Content as success without parsing a body', async () => {
    respond(204)
    await expect(api.unlockOwner('token')).resolves.toBeNull()
  })
})
