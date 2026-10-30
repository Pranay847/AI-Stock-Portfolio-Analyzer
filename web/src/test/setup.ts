import '@testing-library/jest-dom/vitest'

import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Testing Library only auto-unmounts when Vitest globals are on; without this,
// each test's render would pile onto the last one's DOM.
afterEach(() => cleanup())
