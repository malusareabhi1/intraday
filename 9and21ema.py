import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- UI Setup
st.set_page_config(layout="wide")
st.title("📊 EMA 9/21 Crossover Intraday Strategy with Backtest")

# --- Sidebar Inputs
ticker = st.sidebar.text_input("Enter Stock Symbol", value="RELIANCE.NS")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=10))
end_date = st.sidebar.date_input("End Date", datetime.now())
interval = st.sidebar.selectbox("Interval", ['5m', '15m'], index=0)

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
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'Datetime'}, inplace=True)
        else:
            df.insert(0, 'Datetime', pd.to_datetime(df.index))
    return df

def find_column(df, name_part):
    for col in df.columns:
        if name_part.lower() in col.lower() and "adj" not in col.lower():
            return col
    return None

# --- Backtest function for EMA crossover strategy
def backtest_ema_strategy(df, close_col):
    df['EMA9'] = EMAIndicator(close=df[close_col], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df[close_col], window=21).ema_indicator()

    position = None  # None, 'LONG'
    entry_price = 0.0
    trades = []

    for i in range(1, len(df)):
        prev_ema9 = df['EMA9'][i-1]
        prev_ema21 = df['EMA21'][i-1]
        curr_ema9 = df['EMA9'][i]
        curr_ema21 = df['EMA21'][i]
        price = df[close_col][i]
        time = df['Datetime'][i]

        # Generate signals
        if position is None:
            # Entry long when EMA9 crosses above EMA21
            if prev_ema9 < prev_ema21 and curr_ema9 > curr_ema21:
                position = 'LONG'
                entry_price = price
                trades.append({'Entry Time': time, 'Entry Price': price, 'Exit Time': None, 'Exit Price': None, 'Profit': None})
        elif position == 'LONG':
            # Exit long when EMA9 crosses below EMA21
            if prev_ema9 > prev_ema21 and curr_ema9 < curr_ema21:
                position = None
                exit_price = price
                trades[-1]['Exit Time'] = time
                trades[-1]['Exit Price'] = exit_price
                trades[-1]['Profit'] = exit_price - entry_price

    # Close any open position at last price
    if position == 'LONG' and trades:
        trades[-1]['Exit Time'] = df['Datetime'].iloc[-1]
        trades[-1]['Exit Price'] = df[close_col].iloc[-1]
        trades[-1]['Profit'] = trades[-1]['Exit Price'] - entry_price

    return trades, df

def calc_performance(trades):
    if not trades:
        return {}

    total_trades = len(trades)
    wins = sum(1 for t in trades if t['Profit'] and t['Profit'] > 0)
    losses = sum(1 for t in trades if t['Profit'] and t['Profit'] <= 0)
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_profit = sum(t['Profit'] for t in trades if t['Profit'] is not None)
    avg_profit = total_profit / total_trades if total_trades > 0 else 0
    max_drawdown = min(t['Profit'] for t in trades if t['Profit'] is not None) if trades else 0

    return {
        'Total Trades': total_trades,
        'Winning Trades': wins,
        'Losing Trades': losses,
        'Win Rate (%)': f"{win_rate:.2f}",
        'Total Profit': round(total_profit, 2),
        'Avg Profit per Trade': round(avg_profit, 2),
        'Max Drawdown': round(max_drawdown, 2),
    }

# --- Main program flow
df = load_data(ticker, start_date, end_date, interval)

if df.empty:
    st.warning("⚠️ No data available for selected parameters.")
else:
    open_col = find_column(df, "open")
    high_col = find_column(df, "high")
    low_col = find_column(df, "low")
    close_col = find_column(df, "close")

    if None in [open_col, high_col, low_col, close_col]:
        st.error("❌ Required OHLC columns not found in data.")
    else:
        trades, df = backtest_ema_strategy(df, close_col)

        # Plot chart with EMAs and signals
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Datetime'], open=df[open_col], high=df[high_col],
            low=df[low_col], close=df[close_col], name='Candles'))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA9'], line=dict(color='blue'), name='EMA9'))
        fig.add_trace(go.Scatter(x=df['Datetime'], y=df['EMA21'], line=dict(color='orange'), name='EMA21'))

        # Mark entry and exit signals on chart
        for trade in trades:
            if trade['Entry Time'] and trade['Entry Price']:
                fig.add_trace(go.Scatter(
                    x=[trade['Entry Time']], y=[trade['Entry Price']],
                    mode='markers+text',
                    marker=dict(color='green', size=12, symbol='triangle-up'),
                    text=["BUY"], textposition='bottom center'
                ))
            if trade['Exit Time'] and trade['Exit Price']:
                fig.add_trace(go.Scatter(
                    x=[trade['Exit Time']], y=[trade['Exit Price']],
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol='triangle-down'),
                    text=["SELL"], textposition='top center'
                ))

        fig.update_layout(title=f"{ticker} - EMA 9/21 Crossover Strategy with Backtest",
                          height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        # Performance summary
        perf = calc_performance(trades)
        st.subheader("📊 Backtest Performance Metrics")
        if perf:
            st.write(perf)
        else:
            st.info("No trades found during backtest period.")

        # Detailed trades table
        st.subheader("📋 Trades Log")
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df)
