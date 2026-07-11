"""
src/config.py — Central configuration
All environment-based config lives here.
"""
import os

SECRET_KEY         = os.environ.get("QUANTAI_SECRET", "quantai-dev-secret-change-in-production")
PORT               = int(os.environ.get("PORT", 8000))
DB_PATH            = os.environ.get("QUANTAI_DB_PATH", "data/quantai.db")
MODELS_DIR         = os.environ.get("QUANTAI_MODELS_DIR", "models")
TOKEN_EXPIRE_HOURS = int(os.environ.get("QUANTAI_TOKEN_EXPIRE_HOURS", 24))
SIGNALS_CACHE_TTL  = int(os.environ.get("QUANTAI_SIGNALS_CACHE_TTL", 600))  # 10 min
TRI_CACHE_TTL      = int(os.environ.get("QUANTAI_TRI_CACHE_TTL", 1800))     # 30 min

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# NSE market hours (IST, Mon-Fri)
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MIN    = 15
MARKET_CLOSE_HOUR  = 15
MARKET_CLOSE_MIN   = 30
