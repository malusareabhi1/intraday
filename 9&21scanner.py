import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 NIFTY 50 - EMA 9/21 Crossover Scanner")

# ---------------------- Inputs
start_date = datetime.now() - timedelta(days=4)
end_date = datetime.now()
interval = '5m'

# ---------------------- NIFTY 50 List (Editable if needed)
nifty_50 = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS", "ITC.NS", "SBIN.NS",
    "AXISBANK.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "HCLTECH.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "WIPRO.NS",
    "MARUTI.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "TITAN.NS", "ONGC.NS", "COALINDIA.NS", "HINDUNILVR.NS",
    "HINDALCO.NS", "POWERGRID.NS", "JSWSTEEL.NS", "ADANIENT.NS", "DIVISLAB.NS", "NTPC.NS", "GRASIM.NS",
    "TATAMOTORS.NS", "BPCL.NS", "EICHERMOT.NS", "BAJAJFINSV.NS", "TECHM.NS", "HEROMOTOCO.NS", "DRREDDY.NS",
    "BRITANNIA.NS", "BAJAJ-AUTO.NS", "CIPLA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "APOLLOHOSP.NS", "INDUSINDBK.NS",
    "TATASTEEL.NS", "UPL.NS", "SHREECEM.NS", "M&M.NS", "ICICIPRULI.NS", "LTIM.NS"
]

nifty_50 = [
    "ADANIENT.NS", "ADANIPORTS.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AUROPHARMA.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BANDHANBNK.NS", "BANKBARODA.NS", "BEL.NS", "BERGEPAINT.NS",
    "BHARATFORG.NS", "BHARTIARTL.NS", "BIOCON.NS", "BOSCHLTD.NS", "BPCL.NS", "BRITANNIA.NS", "CHOLAFIN.NS",
    "CIPLA.NS", "COALINDIA.NS", "COLPAL.NS", "DABUR.NS", "DIVISLAB.NS", "DLF.NS", "DRREDDY.NS", "EICHERMOT.NS",
    "GAIL.NS", "GODREJCP.NS", "GRASIM.NS", "HAVELLS.NS", "HCLTECH.NS", "HDFC.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ICICIGI.NS", "ICICIPRULI.NS", "IDFCFIRSTB.NS",
    "IGL.NS", "INDIGO.NS", "INDUSINDBK.NS", "INDUSTOWER.NS", "INFY.NS", "IOC.NS", "ITC.NS", "JINDALSTEL.NS",
    "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "LTI.NS", "LUPIN.NS", "M&M.NS", "MARICO.NS", "MARUTI.NS",
    "MCDOWELL-N.NS", "MOTHERSON.NS", "NAUKRI.NS", "NESTLEIND.NS", "NMDC.NS", "NTPC.NS", "ONGC.NS", "PAGEIND.NS",
    "PEL.NS", "PETRONET.NS", "PIIND.NS", "PIDILITIND.NS", "PNB.NS", "POWERGRID.NS", "RECLTD.NS", "RELIANCE.NS",
    "SAIL.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS", "SIEMENS.NS", "SRF.NS", "SUNPHARMA.NS", "TATACHEM.NS",
    "TATACONSUM.NS", "TATAMOTORS.NS", "TATAPOWER.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS", "TITAN.NS", "TORNTPHARM.NS",
    "TRENT.NS", "TVSMOTOR.NS", "UBL.NS", "ULTRACEMCO.NS", "UPL.NS", "VEDL.NS", "VOLTAS.NS", "WIPRO.NS", "ZEEL.NS"
]

# ---------------------- Load & Analyze One Ticker
def get_latest_crossover(symbol):
    try:
        df = yf.download(symbol, start=start_date, end=end_date + timedelta(days=1), interval=interval, progress=False)

        if df.empty or 'Close' not in df.columns:
            return None

        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        close = df['Close']
        df['EMA9'] = EMAIndicator(close=close, window=9).ema_indicator()
        df['EMA21'] = EMAIndicator(close=close, window=21).ema_indicator()

        for i in range(len(df) - 1, 0, -1):
            if df['EMA9'][i-1] < df['EMA21'][i-1] and df['EMA9'][i] > df['EMA21'][i]:
                return {"Symbol": symbol, "Signal": "BUY", "Time": df['Datetime'][i], "Price": df['Close'][i]}
            elif df['EMA9'][i-1] > df['EMA21'][i-1] and df['EMA9'][i] < df['EMA21'][i]:
                return {"Symbol": symbol, "Signal": "SELL", "Time": df['Datetime'][i], "Price": df['Close'][i]}
        return None
    except:
        return None

# ---------------------- Run Scanner
st.info("🔍 Scanning all NIFTY 50 stocks for recent EMA 9/21 crossovers...")

results = []
for symbol in nifty_50:
    res = get_latest_crossover(symbol)
    if res:
        results.append(res)

# ---------------------- Show Results
if results:
    results_df = pd.DataFrame(results).sort_values(by="Time", ascending=False)
    st.success(f"✅ Found {len(results)} recent EMA crossovers")
    st.dataframe(results_df)

    # Optional: Show chart for selected stock
    selected_symbol = st.selectbox("📈 Show Chart for:", results_df["Symbol"].unique())
    if selected_symbol:
        df = yf.download(selected_symbol, start=start_date, end=end_date + timedelta(days=1), interval=interval)
        df.dropna(inplace=True)
        df.reset_index(inplace=True)
        df['EMA9'] = EMAIndicator(close=df['Close'], window=9).ema_indicator()
        df['EMA21'] = EMAIndicator(close=df['Close'], window=21).ema_indicator()

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Datetime'], open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candles'))

        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA9'], line=dict(color='blue'), name='EMA9'))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA21'], line=dict(color='orange'), name='EMA21'))

        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("❌ No recent EMA crossovers found in NIFTY 50 stocks.")
