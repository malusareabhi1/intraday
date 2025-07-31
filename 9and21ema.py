import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from ta.trend import EMAIndicator
from datetime import datetime, timedelta

# ---------------------- UI Setup
st.set_page_config(layout="wide")
st.title("📊 EMA 9/21 Crossover Intraday Strategy")

# ---------------------- Sidebar Inputs
st.sidebar.header("Settings")
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=5))
end_date = st.sidebar.date_input("End Date", datetime.now())
interval = st.sidebar.selectbox("Interval", ['5m', '15m'], index=0)

# ---------------------- Load Data Function
@st.cache_data(ttl=600)
def load_data(symbol, start, end, interval):
    df = yf.download(symbol, start=start, end=end + timedelta(days=1), interval=interval)
    if df.empty:
        return pd.DataFrame()

    # Flatten multi-index if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    # Normalize column names
    df.columns = df.columns.str.strip().str.title()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    # Ensure 'Datetime' column exists
    if 'Datetime' not in df.columns:
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        else:
            df.insert(0, 'Datetime', pd.to_datetime(df.index))

    return df

df = load_data(ticker, start_date, end_date, interval)

# ---------------------- Detect OHLC Columns Safely
def find_column(name_part):
    for col in df.columns:
        if name_part.lower() in col.lower() and "adj" not in col.lower():
            return col
    return None

open_col = find_column("open")
high_col = find_column("high")
low_col  = find_column("low")
close_col = find_column("close")

# ---------------------- Apply Strategy Logic
signals = []
if not df.empty and close_col:
    df['EMA9'] = EMAIndicator(close=df[close_col], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df[close_col], window=21).ema_indicator()

    for i in range(1, len(df)):
        if df['EMA9'][i-1] < df['EMA21'][i-1] and df['EMA9'][i] > df['EMA21'][i]:
            signals.append({'time': df['Datetime'][i], 'price': df[close_col][i], 'type': 'BUY'})
        elif df['EMA9'][i-1] > df['EMA21'][i-1] and df['EMA9'][i] < df['EMA21'][i]:
            signals.append({'time': df['Datetime'][i], 'price': df[close_col][i], 'type': 'SELL'})

# ---------------------- Chart Function
def plot_chart(df, signals, open_col, high_col, low_col, close_col):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Datetime'],
        open=df[open_col], high=df[high_col],
        low=df[low_col], close=df[close_col],
        name='Candles'))

    if 'EMA9' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA9'], line=dict(color='blue'), name='EMA9'))
    if 'EMA21' in df.columns:
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA21'], line=dict(color='orange'), name='EMA21'))

    for s in signals:
        fig.add_trace(go.Scatter(
            x=[s['time']], y=[s['price']],
            mode='markers+text',
            marker=dict(color='green' if s['type'] == 'BUY' else 'red', size=10),
            text=[s['type']], textposition='top center'
        ))

    fig.update_layout(title=f"{ticker} - EMA 9/21 Crossover Strategy", height=600, xaxis_rangeslider_visible=False)
    return fig

# ---------------------- Display Results
if df.empty:
    st.warning("⚠️ No data available for selected parameters.")
elif None in [open_col, high_col, low_col, close_col]:
    st.error("❌ Required OHLC columns not found in data.")
else:
    st.plotly_chart(plot_chart(df, signals, open_col, high_col, low_col, close_col), use_container_width=True)
    st.subheader("📋 Latest Data Snapshot")
    st.dataframe(df.tail())
