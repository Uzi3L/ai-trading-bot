import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

from data_loader import load_config, get_data
from features import add_features
from model import train_model
from trader_alpaca import init_client, check_positions, place_order
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide

import vectorbt as vbt

# Streamlit setup
st.set_page_config(page_title="AI Trading Bot Dashboard", layout="wide")
st.title("🤖 AI Trading Bot Dashboard (Live Mode)")
st.markdown("View live prices, paper trades, and auto-refreshing Alpaca data.")

# Sidebar configuration
cfg = load_config()
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", cfg.get("ticker", "AAPL"))
interval_sec = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 15)

# Initialize Alpaca client
client = init_client(cfg)

# Helper: fetch latest live data from Alpaca
def get_live_data(symbol, lookback_minutes=60):
    data_client = StockHistoricalDataClient(cfg["alpaca"]["api_key"], cfg["alpaca"]["secret_key"])
    now = datetime.utcnow()
    start_time = now - timedelta(minutes=lookback_minutes)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_time,
        end=now
    )
    bars = data_client.get_stock_bars(request).df
    if symbol in bars.index.levels[0]:
        df = bars.loc[symbol]
    else:
        df = bars
    return df

# Layout
col1, col2 = st.columns([3, 1])

# Live mode controls
start_button = st.sidebar.button("▶️ Start Live Stream")
stop_button = st.sidebar.button("⏹️ Stop Stream")

# Session state to control live loop
if "run_live" not in st.session_state:
    st.session_state.run_live = False
if start_button:
    st.session_state.run_live = True
if stop_button:
    st.session_state.run_live = False

# Live stream loop
if st.session_state.run_live:
    st.success(f"📡 Live streaming {ticker} data... updates every {interval_sec}s.")
    placeholder_chart = st.empty()
    placeholder_positions = st.empty()
    last_signal = None

    while st.session_state.run_live:
        try:
            # Fetch recent data
            df = get_live_data(ticker)
            df = add_features(df)
            model = train_model(df)
            features = ['ret_1','ret_5','ma_5','ma_20','vol_20']
            df["Signal"] = model.predict(df[features])
            last_signal = int(df["Signal"].iloc[-1])
            signal_text = "BUY" if last_signal == 1 else "SELL"

            # --- Live price chart ---
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                name="Price"
            ))
            fig.add_trace(go.Scatter(
                x=[df.index[-1]], y=[df['close'].iloc[-1]],
                mode="markers+text",
                text=[f"{signal_text}"],
                textposition="bottom right",
                marker=dict(color="green" if last_signal == 1 else "red", size=12)
            ))
            fig.update_layout(title=f"{ticker} Live Price ({signal_text})", xaxis_rangeslider_visible=False)
            placeholder_chart.plotly_chart(fig, use_container_width=True)

            # --- Refresh live positions ---
            positions = check_positions(client)
            pos_data = []
            for p in positions:
                pos_data.append({
                    "Symbol": p.symbol,
                    "Qty": p.qty,
                    "Market Value": p.market_value,
                    "Unrealized PnL": p.unrealized_pl
                })
            if pos_data:
                placeholder_positions.subheader("📈 Current Paper Positions")
                placeholder_positions.dataframe(pd.DataFrame(pos_data))
            else:
                placeholder_positions.info("No open positions currently.")

            time.sleep(interval_sec)
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            break

else:
    st.info("⏸️ Live stream stopped. Click ▶️ 'Start Live Stream' to begin.")
