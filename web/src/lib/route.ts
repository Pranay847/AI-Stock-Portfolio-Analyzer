import { useCallback, useEffect, useState } from 'react'

export type Route = 'stock' | 'portfolio'

const PATHS: Record<Route, string> = { stock: '/', portfolio: '/portfolio' }

function routeFor(pathname: string): Route {
  return pathname.startsWith('/portfolio') ? 'portfolio' : 'stock'
}

/** Two screens do not need a router library; the server already falls back to
 *  index.html for /portfolio, so deep links and refreshes work. */
export function useRoute(): [Route, (next: Route) => void] {
  const [route, setRoute] = useState<Route>(() => routeFor(window.location.pathname))

  useEffect(() => {
    const onPop = () => setRoute(routeFor(window.location.pathname))
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])

  const navigate = useCallback((next: Route) => {
    if (window.location.pathname !== PATHS[next]) {
      window.history.pushState(null, '', PATHS[next])
    }
    setRoute(next)
  }, [])

  return [route, navigate]
}
