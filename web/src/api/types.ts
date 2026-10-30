// Friendly names for the generated API types. schema.d.ts is produced from the
// FastAPI models by `npm run gen:api` -- never edit it by hand.
import type { components } from './schema'

type Schemas = components['schemas']

export type Capabilities = Schemas['Capabilities']
export type StockAnalysis = Schemas['StockAnalysis']
export type Position = Schemas['Position']
export type PortfolioSummary = Schemas['PortfolioSummary']
export type PortfolioState = Schemas['PortfolioState']
export type SnapTradeState = Schemas['SnapTradeState']
export type ConnectResponse = Schemas['ConnectResponse']
export type PositionAnalysis = Schemas['PositionAnalysis']
export type AnalysisJobState = Schemas['AnalysisJobState']
export type AnalysisStatus = AnalysisJobState['status']
