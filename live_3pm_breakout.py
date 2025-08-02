import time
from datetime import datetime
from broker_zerodha import get_ltp, place_order
from strategy import calculate_quantity  # You already wrote this

symbol = "NIFTY"
offset = 100
capital = 100000
risk_pct = 2

# From previous day 3PM candle
three_pm_high = 22450
three_pm_low = 22390
entry_breakout = three_pm_high + offset
sl = three_pm_low

qty = calculate_quantity(entry_breakout, sl, capital, risk_pct)

print(f"🔍 Watching NIFTY for breakout entry: {entry_breakout} | SL: {sl} | Qty: {qty}")

triggered = False

while not triggered:
    ltp = get_ltp(symbol)
    now = datetime.now().strftime('%H:%M:%S')
    print(f"{now} – LTP: {ltp}")

    if ltp and ltp >= entry_breakout:
        print("🚀 Breakout Triggered!")
        order_id = place_order(symbol="NIFTY23AUG17600CE", qty=qty)
        triggered = True

    time.sleep(30)  # check every 30 seconds
