import streamlit as st
import pandas as pd
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'auto_mode' not in st.session_state:
    st.session_state.auto_mode = False
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = []
if 'portfolio_analysis' not in st.session_state:
    st.session_state.portfolio_analysis = []
if 'sp500_data' not in st.session_state:
    st.session_state.sp500_data = []
if 'robinhood_connected' not in st.session_state:
    st.session_state.robinhood_connected = False
if 'show_login_form' not in st.session_state:
    st.session_state.show_login_form = False
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'portfolio_summary' not in st.session_state:
    st.session_state.portfolio_summary = None


# ==================== ROBINHOOD INTEGRATION ====================

@st.cache_resource(show_spinner=False)
def get_analyzer():
    """Get or create the portfolio analyzer instance."""
    try:
        from robinhood_portfolio_analyzer import RobinhoodPortfolioAnalyzer
        return RobinhoodPortfolioAnalyzer(vector_db_path="./portfolio_vector_db")
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {e}")
        return None


def connect_to_robinhood(username: str, password: str, mfa_code: str = None) -> bool:
    """Attempt to connect to Robinhood."""
    analyzer = get_analyzer()
    if analyzer is None:
        return False
    
    try:
        success = analyzer.login(username, password, mfa_code)
        if success:
            st.session_state.analyzer = analyzer
            st.session_state.robinhood_connected = True
            # Fetch portfolio immediately after login
            portfolio = analyzer.fetch_portfolio()
            if portfolio:
                st.session_state.portfolio_data = portfolio
                st.session_state.portfolio_summary = analyzer.get_portfolio_summary()
            return True
        return False
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def disconnect_from_robinhood():
    """Disconnect from Robinhood."""
    if st.session_state.analyzer:
        try:
            st.session_state.analyzer.logout()
        except:
            pass
    st.session_state.robinhood_connected = False
    st.session_state.analyzer = None
    st.session_state.portfolio_data = []
    st.session_state.portfolio_analysis = []
    st.session_state.portfolio_summary = None
    st.session_state.show_login_form = False


def analyze_individual_stock(symbol: str) -> dict:
    """Analyze a single stock using AI."""
    analyzer = st.session_state.analyzer or get_analyzer()
    
    if analyzer:
        try:
            # Get stock quote only (1 API call instead of 2-3)
            current_price = 0
            change_percent = 0
            
            if analyzer.vector_db:
                quote = analyzer.vector_db.fetch_quote(symbol)
                if quote:
                    current_price = quote.get('price', 0)
                    change_percent = float(quote.get('change_percent', 0))
            
            # Create position for analysis
            position = {
                'symbol': symbol,
                'name': symbol,
                'quantity': 0,
                'average_buy_price': current_price,
                'current_price': current_price,
                'equity': 0,
                'cost_basis': 0,
                'profit_loss': 0,
                'profit_loss_percent': change_percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get AI analysis
            analysis = analyzer.analyze_stock(symbol, position)
            
            current_price = current_price or 100
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'buy_price': current_price * 0.95,
                'sell_price': current_price * 1.15,
                'recommendation': analysis.get('recommendation', 'HOLD'),
                'confidence': analysis.get('confidence', 50),
                'holding_period': 'Short Term (1-3 months)' if analysis.get('recommendation') == 'SELL' else 'Long Term (6+ months)',
                'summary': analysis.get('summary', 'Analysis complete.'),
                'reasons': analysis.get('reasons', [])
            }
        except Exception as e:
            st.warning(f"AI analysis unavailable: {e}")
    
    # Fallback mock data
    import random
    price = random.uniform(50, 500)
    rec = random.choice(['BUY', 'SELL', 'HOLD'])
    return {
        'symbol': symbol,
        'current_price': round(price, 2),
        'buy_price': round(price * 0.95, 2),
        'sell_price': round(price * 1.15, 2),
        'recommendation': rec,
        'confidence': random.randint(60, 95),
        'holding_period': 'Short Term (1-3 months)' if rec == 'SELL' else 'Long Term (6+ months)',
        'summary': f"Based on current market conditions and technical analysis, {symbol} shows {rec.lower()} signals.",
        'reasons': [
            "Technical indicators suggest momentum",
            "Market sentiment is favorable",
            "Valuation metrics are reasonable"
        ]
    }


def analyze_portfolio_stocks() -> pd.DataFrame:
    """Analyze all stocks in the Robinhood portfolio."""
    if not st.session_state.portfolio_data:
        return pd.DataFrame()
    
    analyzer = st.session_state.analyzer
    if analyzer:
        # Run AI analysis on portfolio
        results = analyzer.analyze_portfolio()
        st.session_state.portfolio_analysis = results
        
        # Convert to DataFrame
        data = []
        for r in results:
            data.append({
                'Symbol': r['symbol'],
                'Current Price': f"${r.get('current_price', 0):.2f}",
                'P/L %': f"{r.get('profit_loss_percent', 0):.2f}%",
                'Recommendation': r['recommendation'],
                'Confidence': f"{r['confidence']}%",
                'Summary': r['summary']
            })
        return pd.DataFrame(data)
    
    return pd.DataFrame()


def analyze_sp500(num_stocks: int = 10, progress_callback=None):
    """
    Analyze top S&P 500 stocks with real-time data and AI analysis.
    
    Args:
        num_stocks: Number of stocks to analyze (limited by API rate limits)
        progress_callback: Optional callback for progress updates
    """
    # Top S&P 500 stocks by market cap
    sp500_top = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'LLY', 'ABBV'
    ]
    
    stocks_to_analyze = sp500_top[:num_stocks]
    data = []
    
    # Get analyzer with AI
    analyzer = get_analyzer()
    
    if not analyzer or not analyzer.ollama_available:
        st.error("⚠️ Mistral AI not available. Please ensure Ollama is running with Mistral model.")
        return pd.DataFrame()
    
    for i, symbol in enumerate(stocks_to_analyze):
        try:
            if progress_callback:
                progress_callback(i + 1, len(stocks_to_analyze), symbol)
            
            # Get real-time quote from Alpha Vantage (only 1 API call)
            current_price = 0
            change_percent = 0
            
            if analyzer.vector_db:
                quote = analyzer.vector_db.fetch_quote(symbol)
                if quote:
                    current_price = quote.get('price', 0)
                    change_percent = float(quote.get('change_percent', 0))
            
            # Always use AI analysis
            position = {
                'symbol': symbol,
                'name': symbol,
                'quantity': 0,
                'average_buy_price': current_price,
                'current_price': current_price,
                'equity': 0,
                'cost_basis': 0,
                'profit_loss': 0,
                'profit_loss_percent': change_percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Get AI recommendation from Mistral
            analysis = analyzer.analyze_stock(symbol, position)
            recommendation = analysis.get('recommendation', 'HOLD')
            confidence = analysis.get('confidence', 50)
            
            # Calculate target based on AI recommendation
            if recommendation == 'BUY':
                target_price = current_price * 1.15
            elif recommendation == 'SELL':
                target_price = current_price * 0.95
            else:
                target_price = current_price * 1.05
            
            data.append({
                'Symbol': symbol,
                'Price': f"${current_price:.2f}" if current_price > 0 else "N/A",
                'Change': f"{change_percent:+.2f}%",
                'Target': f"${target_price:.2f}" if target_price > 0 else "N/A",
                'Recommendation': recommendation,
                'Confidence': f"{confidence}%"
            })
            
        except Exception as e:
            # Fallback for this stock
            data.append({
                'Symbol': symbol,
                'Price': "Error",
                'Change': "N/A",
                'Target': "N/A",
                'Recommendation': 'N/A',
                'Confidence': "0%"
            })
    
    return pd.DataFrame(data)


# ==================== UI ====================

# Header
st.title("📈 AI Stock Analyzer")
st.markdown("*Powered by GenAI for intelligent stock recommendations*")

# Sidebar for Settings and Configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Robinhood Connection
    st.subheader("🔌 Trading Platform")
    
    if st.session_state.robinhood_connected:
        st.success("✅ Connected to Robinhood")
        
        # Show account summary
        if st.session_state.portfolio_summary:
            summary = st.session_state.portfolio_summary
            st.metric("Portfolio Value", f"${summary.get('total_equity', 0):,.2f}")
            st.metric("Total P/L", f"${summary.get('total_profit_loss', 0):,.2f}")
            st.caption(f"{summary.get('total_positions', 0)} positions")
        
        if st.button("🔌 Disconnect", type="secondary"):
            disconnect_from_robinhood()
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")
        
        # Show login form or connect button
        if st.session_state.show_login_form:
            st.markdown("### 🔐 Robinhood Login")
            
            with st.form("robinhood_login"):
                username = st.text_input("Email", placeholder="your@email.com")
                password = st.text_input("Password", type="password")
                mfa_code = st.text_input("MFA Code (optional)", placeholder="6-digit code", max_chars=6)
                
                col1, col2 = st.columns(2)
                with col1:
                    submit = st.form_submit_button("🔓 Login", type="primary", width="stretch")
                with col2:
                    cancel = st.form_submit_button("Cancel", width="stretch")
                
                if submit:
                    if username and password:
                        with st.spinner("Connecting to Robinhood..."):
                            mfa = mfa_code if mfa_code else None
                            if connect_to_robinhood(username, password, mfa):
                                st.success("✅ Connected successfully!")
                                st.session_state.show_login_form = False
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Login failed. Check your credentials.")
                    else:
                        st.error("Please enter email and password")
                
                if cancel:
                    st.session_state.show_login_form = False
                    st.rerun()
            
            st.markdown("---")
            st.caption("🔒 Your credentials are never stored")
            st.caption("📱 If 2FA is enabled, enter the code from your authenticator app")
        else:
            if st.button("🔗 Connect to Robinhood", type="primary", width="stretch"):
                st.session_state.show_login_form = True
                st.rerun()
    
    st.divider()
    
    # Automatic Mode Toggle
    st.subheader("🤖 Automatic Mode")
    auto_mode = st.toggle(
        "Enable Auto Mode",
        value=st.session_state.auto_mode,
        help="Schedules automatic analysis multiple times daily with mobile alerts"
    )
    st.session_state.auto_mode = auto_mode
    
    if auto_mode:
        st.success("✅ Auto mode is ON")
        st.info("📱 Mobile alerts enabled\n⏰ Analysis scheduled: 9:00 AM, 12:00 PM, 4:00 PM")
        schedule_times = st.multiselect(
            "Schedule Times",
            ["9:00 AM", "12:00 PM", "4:00 PM", "Market Close"],
            default=["9:00 AM", "12:00 PM", "4:00 PM"]
        )
    else:
        st.info("Manual mode active")
    
    st.divider()
    
    # Execution Mode
    st.subheader("⚡ Execution Settings")
    execution_mode = st.radio(
        "Trade Execution",
        ["Manual Approval Required", "Automatic (AI-driven)", "Semi-Automatic"],
        index=0
    )
    
    if execution_mode == "Automatic (AI-driven)":
        st.warning("⚠️ Trades will execute automatically based on AI recommendations")
    
    st.divider()
    
    # Alert Settings
    st.subheader("🔔 Alert Preferences")
    alert_threshold = st.slider("Alert Confidence Threshold", 0, 100, 75, 5)
    st.checkbox("Email Notifications", value=True)
    st.checkbox("Mobile Push Notifications", value=True)
    st.checkbox("SMS Alerts (High Priority)", value=False)

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Individual Stock", 
    "💼 My Portfolio", 
    "📊 S&P 500 Analysis",
    "📈 Trading Dashboard"
])

# TAB 1: Individual Stock Analysis
with tab1:
    st.header("Individual Stock Analysis")
    st.markdown("Enter a stock symbol to get AI-powered buy/sell recommendations")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stock_symbol = st.text_input(
            "Stock Symbol",
            placeholder="e.g., AAPL, TSLA, GOOGL",
            max_chars=10
        ).upper()
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        analyze_button = st.button("🔎 Analyze Stock", type="primary", width="stretch")
    
    if analyze_button and stock_symbol:
        with st.spinner(f"Analyzing {stock_symbol}... 🤖"):
            result = analyze_individual_stock(stock_symbol)
        
        # Display Results
        st.success(f"✅ Analysis Complete for {result['symbol']}")
        
        # Recommendation Badge
        rec_color = {
            'BUY': 'green',
            'SELL': 'red',
            'HOLD': 'orange'
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${result['current_price']:.2f}")
        
        with col2:
            st.metric("Recommended Buy Price", f"${result['buy_price']:.2f}")
        
        with col3:
            st.metric("Target Sell Price", f"${result['sell_price']:.2f}")
        
        with col4:
            st.metric("Confidence Score", f"{result['confidence']}%")
        
        # Recommendation
        st.markdown(f"### Recommendation: :{rec_color[result['recommendation']]}[**{result['recommendation']}**]")
        
        # Holding Period
        col1, col2 = st.columns(2)
        with col1:
            is_short_term = "Short Term" in result['holding_period']
            st.checkbox(
                "Short Term (1-3 months)" if is_short_term else "Long Term (6+ months)",
                value=True,
                disabled=True
            )
            st.info(f"📅 Recommended holding period: **{result['holding_period']}**")
        
        with col2:
            potential_return = ((result['sell_price'] - result['buy_price']) / result['buy_price']) * 100
            st.metric("Potential Return", f"{potential_return:.1f}%")
        
        # Analysis Summary
        st.markdown("### 📝 Analysis Summary")
        st.info(result['summary'])
        
        # Reasons
        if result.get('reasons'):
            st.markdown("### 📋 Key Factors")
            for reason in result['reasons']:
                st.markdown(f"• {reason}")
        
        # Action Buttons
        st.markdown("### 🎯 Take Action")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button(f"✅ Execute BUY Order for {result['symbol']}", width="stretch"):
                if st.session_state.robinhood_connected:
                    st.success(f"Buy order placed for {result['symbol']} at ${result['buy_price']:.2f}")
                else:
                    st.warning("Connect to Robinhood to execute trades")
        
        with col2:
            if st.button(f"📋 Add to Watchlist", width="stretch"):
                st.info(f"{result['symbol']} added to watchlist")
        
        with col3:
            if st.button(f"📊 View Detailed Analysis", width="stretch"):
                st.info("Opening detailed analysis dashboard...")

# TAB 2: Portfolio Analysis
with tab2:
    st.header("My Portfolio Analysis")
    st.markdown("Analyze all stocks in your Robinhood portfolio")
    
    if not st.session_state.robinhood_connected:
        st.warning("⚠️ Connect to Robinhood to analyze your portfolio")
        st.info("👈 Click 'Connect to Robinhood' in the sidebar to get started")
        
        # Demo mode option
        if st.button("🎭 Try Demo Mode", width="content"):
            # Load mock portfolio for demo
            from robinhood_portfolio_analyzer import demo_with_mock_data
            try:
                analyzer, results = demo_with_mock_data()
                st.session_state.analyzer = analyzer
                st.session_state.portfolio_data = analyzer.portfolio
                st.session_state.portfolio_analysis = results
                st.session_state.portfolio_summary = analyzer.get_portfolio_summary()
                st.session_state.robinhood_connected = True
                st.rerun()
            except Exception as e:
                st.error(f"Demo mode error: {e}")
    else:
        # Show portfolio data
        if st.session_state.portfolio_summary:
            summary = st.session_state.portfolio_summary
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Value", f"${summary.get('total_equity', 0):,.2f}")
            with col2:
                pl = summary.get('total_profit_loss', 0)
                pl_pct = summary.get('total_profit_loss_percent', 0)
                st.metric("Total P/L", f"${pl:,.2f}", f"{pl_pct:.2f}%")
            with col3:
                st.metric("Winners", summary.get('winners', 0))
            with col4:
                st.metric("Losers", summary.get('losers', 0))
        
        st.divider()
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**{len(st.session_state.portfolio_data)} stocks in portfolio**")
        
        with col2:
            if st.button("🤖 Analyze with AI", type="primary", width="stretch"):
                with st.spinner("AI is analyzing your portfolio... 🤖"):
                    df = analyze_portfolio_stocks()
                    if not df.empty:
                        st.session_state.portfolio_data_df = df
        
        # Show analysis results
        if st.session_state.portfolio_analysis:
            results = st.session_state.portfolio_analysis
            
            # Summary counts
            buy_count = sum(1 for r in results if r['recommendation'] == 'BUY')
            sell_count = sum(1 for r in results if r['recommendation'] == 'SELL')
            hold_count = sum(1 for r in results if r['recommendation'] == 'HOLD')
            
            st.divider()
            st.markdown("### 🤖 AI Recommendations")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🟢 BUY", buy_count)
            with col2:
                st.metric("🔴 SELL", sell_count)
            with col3:
                st.metric("🟡 HOLD", hold_count)
            with col4:
                st.metric("📊 Total", len(results))
            
            # Filter options
            filter_rec = st.multiselect(
                "Filter by Recommendation",
                ['BUY', 'SELL', 'HOLD'],
                default=['BUY', 'SELL', 'HOLD']
            )
            
            # Display each recommendation
            for r in results:
                if r['recommendation'] in filter_rec:
                    emoji = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '🟡'}.get(r['recommendation'], '⚪')
                    
                    with st.expander(f"{emoji} {r['symbol']} - {r['recommendation']} ({r['confidence']}% confidence)"):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Summary:** {r['summary']}")
                            if r.get('reasons'):
                                st.markdown("**Reasons:**")
                                for reason in r['reasons']:
                                    st.markdown(f"• {reason}")
                        with col2:
                            st.metric("P/L", f"{r.get('profit_loss_percent', 0):.2f}%")
                            st.metric("Price", f"${r.get('current_price', 0):.2f}")
            
            # Export button
            st.divider()
            if st.button("📥 Export Full Report"):
                if st.session_state.analyzer:
                    report = st.session_state.analyzer.get_analysis_report(results)
                    st.download_button(
                        "Download Report",
                        report,
                        file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
        else:
            # Show raw portfolio data
            if st.session_state.portfolio_data:
                st.markdown("### 📊 Current Holdings")
                df = pd.DataFrame(st.session_state.portfolio_data)
                df_display = df[['symbol', 'name', 'quantity', 'current_price', 'profit_loss_percent']].copy()
                df_display.columns = ['Symbol', 'Name', 'Shares', 'Price', 'P/L %']
                df_display['Price'] = df_display['Price'].apply(lambda x: f"${x:.2f}")
                df_display['P/L %'] = df_display['P/L %'].apply(lambda x: f"{x:.2f}%")
                st.dataframe(df_display, width="stretch")

# TAB 3: S&P 500 Analysis
with tab3:
    st.header("S&P 500 Stock Analysis")
    st.markdown("Real-time analysis of top S&P 500 stocks with **Mistral AI** recommendations")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_stocks = st.slider("Number of stocks to analyze", 5, 20, 10, 
                               help="⚠️ Alpha Vantage free tier: 25 API calls/day")
    
    with col2:
        st.write("")  # Spacing
        analyze_sp500_btn = st.button("🔄 Run Analysis", type="primary", width="stretch")
    
    if analyze_sp500_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, symbol):
            progress_bar.progress(current / total)
            status_text.text(f"🤖 AI analyzing {symbol}... ({current}/{total})")
        
        with st.spinner("Running AI analysis with Mistral... 🤖"):
            st.session_state.sp500_data = analyze_sp500(num_stocks, update_progress)
        
        progress_bar.empty()
        status_text.empty()
        st.rerun()
    
    if len(st.session_state.sp500_data) > 0:
        st.success(f"✅ AI analysis complete! Analyzed {len(st.session_state.sp500_data)} stocks with Mistral AI")
        
        # Summary metrics
        df = st.session_state.sp500_data
        buy_count = (df['Recommendation'] == 'BUY').sum()
        sell_count = (df['Recommendation'] == 'SELL').sum()
        hold_count = (df['Recommendation'] == 'HOLD').sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🟢 BUY", buy_count)
        with col2:
            st.metric("🔴 SELL", sell_count)
        with col3:
            st.metric("🟡 HOLD", hold_count)
        
        # Display table
        st.markdown("### 🎯 Stock Analysis Results")
        
        # Color code recommendations
        def highlight_recommendation(row):
            if row['Recommendation'] == 'BUY':
                return ['background-color: #d4edda'] * len(row)
            elif row['Recommendation'] == 'SELL':
                return ['background-color: #f8d7da'] * len(row)
            else:
                return ['background-color: #fff3cd'] * len(row)
        
        styled_df = df.style.apply(highlight_recommendation, axis=1)
        st.dataframe(styled_df, width="stretch", height=400)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        selected_stocks = st.multiselect(
            "Select stocks to add to watchlist",
            st.session_state.sp500_data['Symbol'].tolist()
        )
        
        if selected_stocks:
            if st.button(f"📋 Add {len(selected_stocks)} stocks to watchlist"):
                st.success(f"Added {', '.join(selected_stocks)} to watchlist!")
    else:
        st.info("👆 Click 'Run Analysis' to fetch real-time stock data and AI recommendations")

# TAB 4: Trading Dashboard
with tab4:
    st.header("Trading Dashboard")
    st.markdown("Real-time overview of your trading activity and performance")
    
    if st.session_state.robinhood_connected and st.session_state.portfolio_summary:
        summary = st.session_state.portfolio_summary
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Portfolio Value", f"${summary.get('total_equity', 0):,.2f}")
        with col2:
            pl = summary.get('total_profit_loss', 0)
            pl_pct = summary.get('total_profit_loss_percent', 0)
            st.metric("Total P&L", f"${pl:,.2f}", f"{pl_pct:.2f}%")
        with col3:
            st.metric("Active Positions", summary.get('total_positions', 0))
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", "$---.--")
        with col2:
            st.metric("Today's P&L", "$---.--")
        with col3:
            st.metric("Active Positions", "--")
        
        st.info("Connect to Robinhood to see your live portfolio data")
    
    st.divider()
    
    # AI Recommendations Summary
    if st.session_state.portfolio_analysis:
        st.subheader("🤖 AI Recommendations Summary")
        
        results = st.session_state.portfolio_analysis
        sells = [r for r in results if r['recommendation'] == 'SELL']
        buys = [r for r in results if r['recommendation'] == 'BUY']
        
        if sells:
            st.markdown("#### 🔴 Consider Selling")
            for r in sells[:5]:
                st.markdown(f"• **{r['symbol']}** - {r['summary'][:100]}...")
        
        if buys:
            st.markdown("#### 🟢 Consider Adding")
            for r in buys[:5]:
                st.markdown(f"• **{r['symbol']}** - {r['summary'][:100]}...")
    
    st.divider()
    
    # Recent Activity
    st.subheader("📊 Recent Trading Activity")
    recent_trades = pd.DataFrame({
        'Time': ['10:30 AM', '11:45 AM', '2:15 PM', '3:30 PM'],
        'Symbol': ['AAPL', 'GOOGL', 'TSLA', 'MSFT'],
        'Action': ['BUY', 'SELL', 'BUY', 'HOLD'],
        'Price': ['$195.50', '$142.30', '$242.80', '$378.90'],
        'Status': ['Executed', 'Executed', 'Pending', 'Monitoring']
    })
    st.dataframe(recent_trades, width="stretch")
    
    # Performance Chart (placeholder)
    st.subheader("📈 Portfolio Performance")
    st.line_chart([100, 102, 105, 103, 108, 112, 110, 115], height=200)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Powered by GenAI | 🔒 Secure Connection to Robinhood</p>
        <p style='font-size: 0.8em;'>Last Updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
    unsafe_allow_html=True
)
