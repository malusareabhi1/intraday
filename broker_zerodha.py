from kiteconnect import KiteConnect
import os

api_key = "your_api_key"
api_secret = "your_api_secret"
access_token = "your_access_token"

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def get_ltp(symbol):
    try:
        data = kite.ltp(f"NSE:{symbol}")
        return data[f"NSE:{symbol}"]['last_price']
    except Exception as e:
        print("Error fetching LTP:", e)
        return None

def place_order(symbol, qty, buy=True, product="MIS"):
    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=symbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY if buy else kite.TRANSACTION_TYPE_SELL,
            quantity=qty,
            order_type=kite.ORDER_TYPE_MARKET,
            product=product
        )
        print("✅ Order Placed. ID:", order_id)
        return order_id
    except Exception as e:
        print("❌ Order Failed:", e)
        return None
