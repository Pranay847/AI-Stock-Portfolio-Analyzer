import os
import json
import pandas as pd
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Check for required packages
try:
    import robin_stocks.robinhood as rh
except ImportError:
    print("❌ robin_stocks not installed. Run: pip install robin_stocks")
    rh = None

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    print("⚠️  ollama not installed. Run: pip install ollama")
    OLLAMA_AVAILABLE = False
    ollama = None

try:
    from alpha_vantage_vector_db import AlphaVantageVectorDB
    VECTOR_DB_AVAILABLE = True
except Exception:
    VECTOR_DB_AVAILABLE = False
    AlphaVantageVectorDB = None

try:
    from models.predict import Predictor
    PREDICTOR_AVAILABLE = True
except Exception:
    PREDICTOR_AVAILABLE = False
    Predictor = None

try:
    from agents.llm_reasoning import generate_rationale, check_llm_available
    LLM_REASONING_AVAILABLE = True
except Exception:
    LLM_REASONING_AVAILABLE = False
    generate_rationale = None

# XGBoost label encoding: 0=SELL, 1=HOLD, 2=BUY
_LABEL_MAP = {0: "SELL", 1: "HOLD", 2: "BUY"}


class RobinhoodPortfolioAnalyzer:
    def __init__(self, vector_db_path: str = "./portfolio_vector_db", use_ollama: bool = True):
        """
        Initialize the analyzer.
        
        Args:
            vector_db_path: Path to store/load the vector database
            use_ollama: If True, use local Mistral via Ollama (free)
        """
        self.is_logged_in = False
        self.portfolio = []
        self.use_ollama = use_ollama
        
        # Initialize Mistral AI via Ollama
        self.ollama_available = False
        if use_ollama and OLLAMA_AVAILABLE:
            try:
                models = ollama.list()
                model_names = [m.get('name', '') for m in models.get('models', [])]
                if any('mistral' in name.lower() for name in model_names):
                    self.ollama_available = True
                    print("✅ Mistral AI ready (via Ollama)")
                else:
                    print("⚠️  Mistral not found. Run: ollama pull mistral")
            except Exception as e:
                print(f"⚠️  Ollama not running: {e}")
        
        # Initialize vector database for RAG
        self.vector_db = None
        if VECTOR_DB_AVAILABLE:
            try:
                self.vector_db = AlphaVantageVectorDB(db_path=vector_db_path)
            except ValueError as e:
                print(f"⚠️  Vector DB init warning: {e}")
        
        print("✅ Robinhood Portfolio Analyzer initialized")
    
    # ==================== ROBINHOOD CONNECTION ====================
    
    def login(self, username: str = None, password: str = None, mfa_code: str = None) -> bool:
        if rh is None:
            print("❌ robin_stocks not installed")
            return False
        
        username = username or os.getenv("ROBINHOOD_USERNAME")
        password = password or os.getenv("ROBINHOOD_PASSWORD")
        
        if not username or not password:
            print("❌ Robinhood credentials not provided")
            print("   Set ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD in .env")
            return False
        
        try:
            try:
                rh.logout()
            except Exception:
                pass

            print("🔐 Logging into Robinhood...")

            if mfa_code:
                login_result = rh.login(username, password, mfa_code=mfa_code)
            else:
                login_result = rh.login(username, password)
            
            if login_result:
                self.is_logged_in = True
                # record which username is currently logged in for clarity
                self.logged_in_user = username
                print("✅ Successfully logged into Robinhood!")
                return True
            else:
                print("❌ Login failed")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return False
    
    def logout(self):
        """Logout from Robinhood."""
        if rh:
            rh.logout()
            self.is_logged_in = False
            # clear tracked username when logging out
            try:
                self.logged_in_user = None
            except Exception:
                pass
            print("👋 Logged out of Robinhood")
    
    # ==================== PORTFOLIO FETCHING ====================
    
    def fetch_portfolio(self) -> list[dict]:
        """
        Fetch all stock holdings from Robinhood portfolio.
        
        Returns:
            List of portfolio positions
        """
        if not self.is_logged_in:
            print("❌ Not logged in to Robinhood")
            return []
        
        try:
            print("📊 Fetching portfolio...")
            
            # Get all stock positions
            positions = rh.account.get_open_stock_positions()
            
            portfolio = []
            for position in positions:
                # Get instrument details
                instrument_url = position.get('instrument')
                instrument = rh.stocks.get_instrument_by_url(instrument_url)
                symbol = instrument.get('symbol', 'UNKNOWN')
                
                # Get current quote
                quote = rh.stocks.get_latest_price(symbol)
                current_price = float(quote[0]) if quote and quote[0] else 0
                
                # Calculate position details
                quantity = float(position.get('quantity', 0))
                avg_buy_price = float(position.get('average_buy_price', 0))
                equity = quantity * current_price
                cost_basis = quantity * avg_buy_price
                profit_loss = equity - cost_basis
                profit_loss_pct = ((current_price - avg_buy_price) / avg_buy_price * 100) if avg_buy_price > 0 else 0
                
                portfolio.append({
                    'symbol': symbol,
                    'name': instrument.get('simple_name', symbol),
                    'quantity': quantity,
                    'average_buy_price': avg_buy_price,
                    'current_price': current_price,
                    'equity': equity,
                    'cost_basis': cost_basis,
                    'profit_loss': profit_loss,
                    'profit_loss_percent': profit_loss_pct,
                    'timestamp': datetime.now().isoformat()
                })
            
            self.portfolio = portfolio
            print(f"✅ Found {len(portfolio)} positions in portfolio")
            return portfolio
            
        except Exception as e:
            print(f"❌ Error fetching portfolio: {str(e)}")
            return []
    
    def get_portfolio_summary(self) -> dict:
        """Get summary statistics for the portfolio."""
        if not self.portfolio:
            return {}
        
        total_equity = sum(p['equity'] for p in self.portfolio)
        total_cost = sum(p['cost_basis'] for p in self.portfolio)
        total_pl = sum(p['profit_loss'] for p in self.portfolio)
        total_pl_pct = ((total_equity - total_cost) / total_cost * 100) if total_cost > 0 else 0
        
        winners = [p for p in self.portfolio if p['profit_loss'] > 0]
        losers = [p for p in self.portfolio if p['profit_loss'] < 0]
        
        return {
            'total_positions': len(self.portfolio),
            'total_equity': total_equity,
            'total_cost_basis': total_cost,
            'total_profit_loss': total_pl,
            'total_profit_loss_percent': total_pl_pct,
            'winners': len(winners),
            'losers': len(losers),
            'best_performer': max(self.portfolio, key=lambda x: x['profit_loss_percent'])['symbol'] if self.portfolio else None,
            'worst_performer': min(self.portfolio, key=lambda x: x['profit_loss_percent'])['symbol'] if self.portfolio else None
        }
    
    # ==================== STOCK DATA & CONTEXT ====================
    
    def get_stock_context(self, symbol: str, use_vector_db: bool = True) -> str:
        """
        Get comprehensive context for a stock for AI analysis.
        
        Args:
            symbol: Stock ticker symbol
            use_vector_db: Whether to use vector DB for additional context
            
        Returns:
            Context string for AI analysis
        """
        # Clean and validate symbol
        symbol = symbol.strip().upper() if symbol else ""
        if not symbol:
            return ""
        
        context_parts = []
        
        # Get portfolio position info
        position = next((p for p in self.portfolio if p['symbol'] == symbol), None)
        if position:
            context_parts.append(f"""
PORTFOLIO POSITION:
Symbol: {position['symbol']}
Name: {position['name']}
Shares Owned: {position['quantity']:.4f}
Average Buy Price: ${position['average_buy_price']:.2f}
Current Price: ${position['current_price']:.2f}
Total Equity: ${position['equity']:.2f}
Profit/Loss: ${position['profit_loss']:.2f} ({position['profit_loss_percent']:.2f}%)
            """.strip())
        
        # Get additional context from vector DB
        if use_vector_db and self.vector_db:
            try:
                # Try to get stored context
                db_context = self.vector_db.get_stock_context(symbol)
                if db_context:
                    context_parts.append(f"\nMARKET DATA:\n{db_context}")
                else:
                    # Fetch fresh data if not in DB
                    print(f"   📡 Fetching fresh data for {symbol}...")
                    result = self.vector_db.process_stock(symbol, include_news=False)
                    if result.get('success'):
                        db_context = self.vector_db.get_stock_context(symbol)
                        if db_context:
                            context_parts.append(f"\nMARKET DATA:\n{db_context}")
            except Exception as e:
                print(f"   ⚠️  Could not get vector DB context: {e}")
        
        return "\n\n".join(context_parts)
    
    # ==================== AI ANALYSIS ====================
    
    def analyze_stock(self, symbol: str, position: dict = None) -> dict:
        """
        Analyze a single stock and get AI recommendation.
        
        Args:
            symbol: Stock ticker symbol
            position: Optional position data (will fetch from portfolio if not provided)
            
        Returns:
            Analysis result with recommendation
        """
        if not position:
            position = next((p for p in self.portfolio if p['symbol'] == symbol), None)
        
        if not position:
            return {
                'symbol': symbol,
                'recommendation': 'UNKNOWN',
                'confidence': 0,
                'summary': 'No position data available',
                'reasons': []
            }
        
        # Get comprehensive context from vector DB (RAG)
        context = self.get_stock_context(symbol)

        # Use AI analysis if either LangChain reasoning or Ollama is available
        if LLM_REASONING_AVAILABLE or self.ollama_available:
            return self._ai_analysis(position, context)

        return self._rule_based_analysis(position)
    
    def _rule_based_analysis(self, position: dict) -> dict:
        """
        Simple rule-based analysis when AI is not available.
        """
        pl_pct = position['profit_loss_percent']
        
        if pl_pct > 20:
            recommendation = 'SELL'
            confidence = 70
            summary = f"Consider taking profits. {position['symbol']} is up {pl_pct:.1f}%."
            reasons = [
                f"Stock is up {pl_pct:.1f}% from your buy price",
                "Taking profits locks in gains",
                "Consider rebalancing your portfolio"
            ]
        elif pl_pct < -15:
            recommendation = 'HOLD'
            confidence = 60
            summary = f"Consider holding or averaging down. {position['symbol']} is down {abs(pl_pct):.1f}%."
            reasons = [
                f"Stock is down {abs(pl_pct):.1f}% - selling now locks in losses",
                "Consider if the fundamentals have changed",
                "Averaging down may lower your cost basis"
            ]
        else:
            recommendation = 'HOLD'
            confidence = 65
            summary = f"Hold position. {position['symbol']} is performing within normal range."
            reasons = [
                "Position is within acceptable profit/loss range",
                "No strong signals for immediate action",
                "Continue monitoring for changes"
            ]
        
        return {
            'symbol': position['symbol'],
            'recommendation': recommendation,
            'confidence': confidence,
            'summary': summary,
            'reasons': reasons,
            'current_price': position['current_price'],
            'profit_loss_percent': pl_pct,
            'analysis_type': 'rule_based'
        }
    
    def _get_xgboost_prediction(self, ticker: str):
        """Load pre-computed processed data and return XGBoost (signal, confidence)."""
        if not PREDICTOR_AVAILABLE:
            return None, None
        try:
            processed_path = os.path.join("data", "processed", f"{ticker}.csv")
            if not os.path.exists(processed_path):
                return None, None
            df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
            exclude = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                       'future_close', 'future_return', 'label', 'ticker'}
            features = [c for c in df.columns
                        if c not in exclude and df[c].dtype in ('float64', 'int64')]
            if not features:
                return None, None
            X_latest = df[features].tail(1)
            predictor = Predictor()
            label, confidence, _ = predictor.predict_for_ticker(ticker, X_latest)
            signal = _LABEL_MAP.get(int(label), "HOLD")
            return signal, round(float(confidence) * 100, 1)
        except Exception as e:
            print(f"   ⚠️  XGBoost prediction error for {ticker}: {e}")
            return None, None

    def _ai_analysis(self, position: dict, context: str) -> dict:
        """AI-powered analysis: XGBoost signal + LangChain LLM reasoning."""
        ticker = position['symbol']

        portfolio_status = (
            f"Owned: {'Yes' if position['quantity'] > 0 else 'No'} | "
            f"Shares: {position['quantity']:.4f} | "
            f"Avg Buy: ${position['average_buy_price']:.2f} | "
            f"Current: ${position['current_price']:.2f} | "
            f"P/L: {position['profit_loss_percent']:.2f}%"
        )
        quote_data = (
            f"Price: ${position['current_price']:.2f} | "
            f"Equity: ${position['equity']:.2f} | "
            f"Cost Basis: ${position['cost_basis']:.2f}"
        )

        xgb_signal, xgb_confidence = self._get_xgboost_prediction(ticker)

        if LLM_REASONING_AVAILABLE:
            result = generate_rationale(
                ticker=ticker,
                portfolio_status=portfolio_status,
                quote_data=quote_data,
                xgboost_signal=xgb_signal or "No model prediction available",
                xgboost_confidence=xgb_confidence or 0,
                rag_context=context,
            )
        else:
            result = self._ollama_fallback(position, context)

        result['symbol'] = ticker
        result.setdefault('current_price', position['current_price'])
        result.setdefault('profit_loss_percent', position['profit_loss_percent'])
        if xgb_signal:
            result['xgboost_signal'] = xgb_signal
            result['xgboost_confidence'] = xgb_confidence
        return result

    def _ollama_fallback(self, position: dict, context: str) -> dict:
        """Direct Ollama call used only when langchain LLM reasoning is unavailable."""
        import re
        prompt = (
            f"You are an expert financial analyst.\n\n{context}\n\n"
            "Respond in JSON: {\"recommendation\": \"BUY|SELL|HOLD\", "
            "\"confidence\": 75, \"summary\": \"...\", \"reasons\": [], \"risks\": []}"
        )
        try:
            response = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": "You are a financial analyst AI. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            result = json.loads(json_match.group() if json_match else content)
            result.setdefault('risks', [])
            result['analysis_type'] = 'mistral_ai'
            return result
        except Exception as e:
            print(f"   ⚠️  Ollama fallback error: {e}")
            return self._rule_based_analysis(position)
    
    def analyze_portfolio(self, max_stocks: int = None) -> list[dict]:
        """
        Analyze all stocks in portfolio.
        
        Args:
            max_stocks: Maximum number of stocks to analyze (for API limits)
            
        Returns:
            List of analysis results
        """
        if not self.portfolio:
            print("❌ No portfolio data. Call fetch_portfolio() first.")
            return []
        
        stocks_to_analyze = self.portfolio[:max_stocks] if max_stocks else self.portfolio
        
        print(f"\n{'='*60}")
        print(f"🤖 ANALYZING {len(stocks_to_analyze)} STOCKS")
        print('='*60)
        
        results = []
        for i, position in enumerate(stocks_to_analyze, 1):
            symbol = position['symbol']
            print(f"\n[{i}/{len(stocks_to_analyze)}] Analyzing {symbol}...")
            
            result = self.analyze_stock(symbol, position)
            results.append(result)
            
            # Print quick summary
            rec = result['recommendation']
            emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(rec, '⚪')
            print(f"   {emoji} {rec} (confidence: {result['confidence']}%)")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print('='*60)
        
        buy_count = sum(1 for r in results if r['recommendation'] == 'BUY')
        sell_count = sum(1 for r in results if r['recommendation'] == 'SELL')
        hold_count = sum(1 for r in results if r['recommendation'] == 'HOLD')
        
        print(f"   🟢 BUY:  {buy_count} stocks")
        print(f"   🔴 SELL: {sell_count} stocks")
        print(f"   🟡 HOLD: {hold_count} stocks")
        
        return results
    
    def get_analysis_report(self, results: list[dict]) -> str:
        """
        Generate a formatted analysis report.
        
        Args:
            results: List of analysis results
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("📈 PORTFOLIO ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Group by recommendation
        buys = [r for r in results if r['recommendation'] == 'BUY']
        sells = [r for r in results if r['recommendation'] == 'SELL']
        holds = [r for r in results if r['recommendation'] == 'HOLD']
        
        if sells:
            report.append("\n🔴 SELL RECOMMENDATIONS:")
            report.append("-" * 40)
            for r in sells:
                report.append(f"\n{r['symbol']} - Confidence: {r['confidence']}%")
                report.append(f"   P/L: {r['profit_loss_percent']:.2f}%")
                report.append(f"   {r['summary']}")
                for reason in r.get('reasons', []):
                    report.append(f"   • {reason}")
        
        if buys:
            report.append("\n🟢 BUY RECOMMENDATIONS (Add to position):")
            report.append("-" * 40)
            for r in buys:
                report.append(f"\n{r['symbol']} - Confidence: {r['confidence']}%")
                report.append(f"   P/L: {r['profit_loss_percent']:.2f}%")
                report.append(f"   {r['summary']}")
                for reason in r.get('reasons', []):
                    report.append(f"   • {reason}")
        
        if holds:
            report.append("\n🟡 HOLD RECOMMENDATIONS:")
            report.append("-" * 40)
            for r in holds:
                report.append(f"\n{r['symbol']} - Confidence: {r['confidence']}%")
                report.append(f"   P/L: {r['profit_loss_percent']:.2f}%")
                report.append(f"   {r['summary']}")
        
        report.append("\n" + "=" * 70)
        report.append("⚠️  DISCLAIMER: This is not financial advice. Always do your own research.")
        report.append("=" * 70)
        
        return "\n".join(report)


# ==================== DEMO / TESTING ====================

def demo_with_mock_data():
    """
    Demo the analyzer with mock portfolio data (no Robinhood login required).
    """
    print("\n" + "="*70)
    print("🎭 DEMO MODE - Using Mock Portfolio Data")
    print("="*70)
    
    # Create analyzer
    analyzer = RobinhoodPortfolioAnalyzer(vector_db_path="./demo_vector_db")
    
    # Mock portfolio data
    analyzer.portfolio = [
        {
            'symbol': 'AAPL',
            'name': 'Apple Inc',
            'quantity': 10,
            'average_buy_price': 150.00,
            'current_price': 175.00,
            'equity': 1750.00,
            'cost_basis': 1500.00,
            'profit_loss': 250.00,
            'profit_loss_percent': 16.67,
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'GOOGL',
            'name': 'Alphabet Inc',
            'quantity': 5,
            'average_buy_price': 140.00,
            'current_price': 135.00,
            'equity': 675.00,
            'cost_basis': 700.00,
            'profit_loss': -25.00,
            'profit_loss_percent': -3.57,
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'TSLA',
            'name': 'Tesla Inc',
            'quantity': 8,
            'average_buy_price': 200.00,
            'current_price': 250.00,
            'equity': 2000.00,
            'cost_basis': 1600.00,
            'profit_loss': 400.00,
            'profit_loss_percent': 25.00,
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'NVDA',
            'name': 'NVIDIA Corporation',
            'quantity': 3,
            'average_buy_price': 450.00,
            'current_price': 480.00,
            'equity': 1440.00,
            'cost_basis': 1350.00,
            'profit_loss': 90.00,
            'profit_loss_percent': 6.67,
            'timestamp': datetime.now().isoformat()
        }
    ]
    
    analyzer.is_logged_in = True  # Pretend we're logged in
    
    # Show portfolio summary
    summary = analyzer.get_portfolio_summary()
    print("\n📊 PORTFOLIO SUMMARY:")
    print(f"   Total Positions: {summary['total_positions']}")
    print(f"   Total Equity: ${summary['total_equity']:,.2f}")
    print(f"   Total P/L: ${summary['total_profit_loss']:,.2f} ({summary['total_profit_loss_percent']:.2f}%)")
    print(f"   Winners: {summary['winners']} | Losers: {summary['losers']}")
    
    # Analyze portfolio
    results = analyzer.analyze_portfolio()
    
    # Generate and print report
    report = analyzer.get_analysis_report(results)
    print(report)
    
    return analyzer, results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 ROBINHOOD PORTFOLIO ANALYZER")
    print("="*70)
    
    print("\nChoose mode:")
    print("1. Demo with mock data (no login required)")
    print("2. Connect to Robinhood (requires credentials)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        # Real Robinhood connection
        analyzer = RobinhoodPortfolioAnalyzer()
        
        username = input("Robinhood username/email: ").strip()
        password = input("Robinhood password: ").strip()
        mfa = input("MFA code (or press Enter to skip): ").strip() or None
        
        if analyzer.login(username, password, mfa):
            portfolio = analyzer.fetch_portfolio()
            if portfolio:
                results = analyzer.analyze_portfolio()
                report = analyzer.get_analysis_report(results)
                print(report)
            analyzer.logout()
    else:
        # Demo mode
        demo_with_mock_data()
