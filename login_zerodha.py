from kiteconnect import KiteConnect
import webbrowser

api_key = "your_api_key"
api_secret = "your_api_secret"

kite = KiteConnect(api_key=api_key)
login_url = kite.login_url()
print("🔗 Login URL:", login_url)

# Open login in browser
webbrowser.open(login_url)

# After login, user gets request_token in URL, you need to paste it manually
request_token = input("Enter request token from URL: ")

data = kite.generate_session(request_token, api_secret=api_secret)
access_token = data["access_token"]
print("✅ Access Token:", access_token)
