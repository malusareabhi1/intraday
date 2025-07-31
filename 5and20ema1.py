import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- NIFTY 50 list
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

# --- UI Setup
st.set_page_config(layout="wide")
st.title("📈 MA Crossover + Volume Confirmation Intraday Strategy")

# --- Sidebar inputs
st.sidebar.header("Settings")
mode = st.sidebar.radio("Select Mode", ["Single Stock", "Scan NIFTY 50"])
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=10))
end_date = st.sidebar.date_input("End Date", datetime.now())
interval = st.sidebar.selectbox("Interval", ['5m', '15m'], index=0)

if mode == "Single Stock":
    ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
else:
    ticker = st.sidebar.selectbox("Select Stock from NIFTY 50", nifty_50)

# --- Data loader
@st.cache_data(ttl=600)
def load_data(symbol, start, end, interval):
    df = yf.download(symbol, start=start, end=end + timedelta(days=1), interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]
    df.columns = df.columns.str.strip().str.title()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    if 'Datetime' not in df.columns:
        df.insert(0, 'Datetime', pd.to_datetime(df.index))
    return df

def find_column(df, name_part):
    for col in df.columns:
        if name_part.lower() in col.lower() and "adj" not in col.lower():
            return col
    return None

def backtest_ma_volume_strategy(df, close_col, volume_col):
    df['EMA5'] = EMAIndicator(close=df[close_col], window=5).ema_indicator()
    df['EMA20'] = EMAIndicator(close=df[close_col], window=20).ema_indicator()
    df['AvgVol20'] = df[volume_col].rolling(window=20).mean()
    df['EMA5_prev'] = df['EMA5'].shift(1)
    df['EMA20_prev'] = df['EMA20'].shift(1)

    buy_signals = (df['EMA5_prev'] < df['EMA20_prev']) & (df['EMA5'] > df['EMA20']) & (df[volume_col] > df['AvgVol20'])
    sell_signals = (df['EMA5_prev'] > df['EMA20_prev']) & (df['EMA5'] < df['EMA20'])

    df['Signal'] = 0
    df.loc[buy_signals, 'Signal'] = 1
    df.loc[sell_signals, 'Signal'] = -1

    trades = []
    position = None
    entry_index = None

    for i, row in df.iterrows():
        signal = row['Signal']
        price = row[close_col]
        time = row['Datetime']

        if position is None and signal == 1:
            position = 'LONG'
            entry_index = i
        elif position == 'LONG' and signal == -1:
            entry_price = df.at[entry_index, close_col]
            exit_price = price
            trades.append({
                'Entry Time': df.at[entry_index, 'Datetime'],
                'Entry Price': entry_price,
                'Exit Time': time,
                'Exit Price': exit_price,
                'Profit': exit_price - entry_price
            })
            position = None
            entry_index = None

    if position == 'LONG' and entry_index is not None:
        entry_price = df.at[entry_index, close_col]
        exit_price = df.iloc[-1][close_col]
        trades.append({
            'Entry Time': df.at[entry_index, 'Datetime'],
            'Entry Price': entry_price,
            'Exit Time': df.iloc[-1]['Datetime'],
            'Exit Price': exit_price,
            'Profit': exit_price - entry_price
        })

    return trades, df

def calc_performance(trades):
    if not trades:
        return {}
    total_trades = len(trades)
    wins = sum(1 for t in trades if t['Profit'] > 0)
    losses = total_trades - wins
    return {
        'Total Trades': total_trades,
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Win Rate (%)': f"{wins / total_trades * 100:.2f}",
        'Total Profit': round(sum(t['Profit'] for t in trades), 2),
        'Avg Profit per Trade': round(sum(t['Profit'] for t in trades) / total_trades, 2),
        'Max Drawdown': round(min(t['Profit'] for t in trades), 2),
    }

def plot_strategy_chart(df, trades, open_col, high_col, low_col, close_col, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['Datetime'], open=df[open_col], high=df[high_col],
        low=df[low_col], close=df[close_col], name='Candles'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA5'], line=dict(color='blue'), name='EMA5'))
    fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA20'], line=dict(color='orange'), name='EMA20'))

    for trade in trades:
        fig.add_trace(go.Scatter(
            x=[trade['Entry Time']], y=[trade['Entry Price']],
            mode='markers+text',
            marker=dict(color='green', size=10), text=["BUY"], textposition="bottom center"))
        fig.add_trace(go.Scatter(
            x=[trade['Exit Time']], y=[trade['Exit Price']],
            mode='markers+text',
            marker=dict(color='red', size=10), text=["SELL"], textposition="top center"))

    fig.update_layout(title=f"{ticker} - MA + Volume Strategy", height=600, xaxis_rangeslider_visible=False)
    return fig

# === Main Execution ===
if mode == "Single Stock":
    df = load_data(ticker, start_date, end_date, interval)
    if df.empty:
        st.warning("No data for selected stock.")
    else:
        open_col = find_column(df, "open")
        high_col = find_column(df, "high")
        low_col = find_column(df, "low")
        close_col = find_column(df, "close")
        volume_col = find_column(df, "volume")

        if None in [open_col, high_col, low_col, close_col, volume_col]:
            st.error("Required columns missing.")
        else:
            trades, df = backtest_ma_volume_strategy(df, close_col, volume_col)
            st.plotly_chart(plot_strategy_chart(df, trades, open_col, high_col, low_col, close_col, ticker), use_container_width=True)
            st.subheader("📊 Performance Metrics")
            st.write(calc_performance(trades))
            st.subheader("📋 Trade Log")
            st.dataframe(pd.DataFrame(trades))

else:
    st.subheader("📊 Scanning NIFTY 50 Stocks...")
    if st.button("Run Scan"):
        results = []
        all_trade_logs = []
        progress = st.progress(0)

        for i, symbol in enumerate(nifty_50):
            df = load_data(symbol, start_date, end_date, interval)
            if df.empty:
                continue
            open_col = find_column(df, "open")
            high_col = find_column(df, "high")
            low_col = find_column(df, "low")
            close_col = find_column(df, "close")
            volume_col = find_column(df, "volume")
            if None in [open_col, high_col, low_col, close_col, volume_col]:
                continue
            trades, _ = backtest_ma_volume_strategy(df, close_col, volume_col)
            perf = calc_performance(trades)
            if perf:
                perf['Symbol'] = symbol
                results.append(perf)
            for t in trades:
                t['Symbol'] = symbol
                all_trade_logs.append(t)
            progress.progress((i + 1) / len(nifty_50))

        if results:
            result_df = pd.DataFrame(results).sort_values(by="Total Profit", ascending=False).reset_index(drop=True)
            st.subheader("📈 Top Performing Stocks")
            st.dataframe(result_df)

            csv_summary = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Summary Report", csv_summary, "nifty50_summary.csv", "text/csv")

            if all_trade_logs:
                csv_trades = pd.DataFrame(all_trade_logs).to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download All Trade Logs", csv_trades, "nifty50_trade_logs.csv", "text/csv")
        else:
            st.info("No profitable trades found.")
