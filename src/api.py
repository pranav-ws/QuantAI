"""
src/api.py — QuantAI FastAPI backend v4.0
New in v4: Multi-user auth, watchlist, per-user portfolio, admin panel, scheduler status.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException, Header, Depends# type: ignore
from fastapi.middleware.cors import CORSMiddleware# type: ignore
from pydantic import BaseModel# type: ignore
from datetime import datetime, date# type: ignore
import sqlite3, joblib, json, pandas as pd# type: ignore
from src.features import get_feature_dataset, add_features
from src.model import FEATURE_COLS
from src.data_collector import STOCK_UNIVERSE
import yfinance as yf# type: ignore


from src.options_pricing import (
    bsm_price, bsm_greeks, implied_volatility,
    get_lot_size, RISK_FREE_RATE,
)
from src.options_chain import fetch_full_chain, build_iv_surface
from src.options_strategies import (
    suggest_strategy, long_call, long_put,
    bull_call_spread, bear_put_spread,
    long_straddle, iron_condor,
)

app = FastAPI(title="QuantAI API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join("data", "quantai.db")

# ── Auth dependency ───────────────────────────────────────

def get_current_user(authorization: str = Header(default="")):
    """
    Extracts Bearer token from Authorization header and returns user_id.
    Use in endpoints that need auth.
    """
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Authorization token missing")
    from src.auth import validate_token
    uid = validate_token(token)
    if uid is None:
        raise HTTPException(status_code=401, detail="Token invalid or expired")
    return uid

def get_current_user_optional(authorization: str = Header(default="")):
    """Same as get_current_user but returns None instead of 401 — for public endpoints."""
    token = authorization.replace("Bearer ", "").strip()
    if not token:
        return None
    from src.auth import validate_token
    return validate_token(token)

# ── ML Signal helper ──────────────────────────────────────

def get_signal_for(ticker):
    try:
        df = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
        if df.empty:
            return None
        if hasattr(df.columns, 'levels'):
            df.columns = [col[0] for col in df.columns]
        df = add_features(df)
        if df.empty or len(df) < 30:
            return None

        from src.ensemble_model import get_ensemble_signal_full, get_model_agreement
        ens_conf, individual, models_used, rb_details, ml_count, rb_count = \
            get_ensemble_signal_full(ticker, df)

        if ens_conf is None:
            return None

        latest    = df.iloc[-1]
        price     = float(latest["Close"])
        signal    = "BUY" if ens_conf >= 0.58 else "SKIP"
        agreement = get_model_agreement(individual) if individual else "single"

        # Build a concise rule-based summary for the API response —
        # only strategies that actually fired (rb_details is already filtered)
        rule_based_summary = {
            name: {
                "signal"    : d["signal"],
                "confidence": d["confidence"],
                "reason"    : d["reason"],
            }
            for name, d in rb_details.items()
        }

        # model_type string e.g. "Ensemble(3+2)" — 3 ML, 2 rule-based
        model_type = (f"Ensemble({ml_count}+{rb_count})" if rb_count > 0
                      else f"Ensemble({ml_count})")

        return {
            "ticker"        : ticker,
            "name"          : STOCK_UNIVERSE[ticker][0],
            "sector"        : STOCK_UNIVERSE[ticker][1],
            "price"         : round(price, 2),
            "confidence"    : round(ens_conf, 4),
            "signal"        : signal,
            "model_type"    : model_type,
            "models_used"   : models_used,
            "ml_count"      : ml_count,
            "rb_count"      : rb_count,
            "agreement"     : agreement,
            "individual"    : {k: round(v, 4) for k, v in (individual or {}).items()},
            "rule_based"    : rule_based_summary,
            "rsi"           : round(float(latest["RSI_14"]), 2),
            "macd"          : round(float(latest["MACD"]), 4),
            "bb_width"      : round(float(latest["BB_Width"]), 4),
            "volume_ratio"  : round(float(latest["Volume_Ratio"]), 2),
            "updated_at"    : datetime.now().isoformat(),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ═══════════════════════════════════════════════════════════
#  PUBLIC ROUTES (no auth needed)
# ═══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "QuantAI API running", "version": "4.0",
            "endpoints": ["/signals", "/signal/{ticker}",
                          "/news/{ticker}", "/portfolio", "/prices/{ticker}",
                          "/auth/register", "/auth/login", "/auth/me",
                          "/watchlist", "/user/portfolio", "/admin/users"]}

@app.get("/signals")
def get_all_signals(user_id: int = Depends(get_current_user_optional)):
    """Scans all 50 Nifty stocks and returns ensemble ML signals."""
    results = []
    for ticker in STOCK_UNIVERSE:
        sig = get_signal_for(ticker)
        if sig:
            results.append(sig)
    buys = [r for r in results if r.get("signal") == "BUY"]
    return {
        "scan_time"   : datetime.now().isoformat(),
        "total_stocks": len(results),
        "buy_signals" : len(buys),
        "results"     : sorted(results,
                               key=lambda x: x.get("confidence", 0),
                               reverse=True)
    }

@app.get("/signal/{ticker}")
def get_single_signal(ticker: str):
    """Returns ensemble signal for one specific ticker."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    sig = get_signal_for(t)
    if not sig:
        return {"error": f"No model found for {t}"}
    return sig

@app.get("/news/{ticker}")
def get_news(ticker: str, limit: int = 8):
    """Returns recent news headlines with VADER sentiment scores."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    try:
        from src.news_sentiment import get_sentiment, sentiment_label, get_sentiment_modifier
        score, articles = get_sentiment(t, max_articles=limit)
        modifier        = get_sentiment_modifier(score)
        return {
            "ticker"          : t,
            "sentiment_score" : score,
            "sentiment_label" : sentiment_label(score),
            "modifier"        : modifier,
            "articles"        : articles[:limit],
        }
    except Exception as e:
        return {"ticker": t, "error": str(e), "articles": []}

@app.get("/prices/{ticker}")
def get_prices(ticker: str, days: int = 90):
    """Returns historical OHLCV prices from database."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume "
        "FROM prices WHERE ticker=? ORDER BY date DESC LIMIT ?",
        conn, params=(t, days)
    )
    conn.close()
    return {"ticker": t, "days": len(df), "prices": df.to_dict(orient="records")}

@app.get("/portfolio")
def get_portfolio():
    """Returns global paper trading portfolio state."""
    trades_path  = os.path.join("data", "paper_trades.json")
    capital_path = os.path.join("data", "paper_capital.json")
    cap    = json.load(open(capital_path)) if os.path.exists(capital_path) else {}
    trades = json.load(open(trades_path))  if os.path.exists(trades_path)  else []
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    open_t = [t for t in trades if t.get("status") == "OPEN"]
    return {
        "capital"      : cap.get("capital", 100000),
        "start_date"   : cap.get("start", str(date.today())),
        "total_trades" : len(trades),
        "open_trades"  : len(open_t),
        "closed_trades": len(closed),
        "trade_log"    : trades[-20:]
    }

# ═══════════════════════════════════════════════════════════
#  correlation routes
# ═══════════════════════════════════════════════════════════


@app.get("/correlation")
async def get_correlation(period: str = "2y", current_user=Depends(get_current_user)):
    """
    Returns pairwise correlations and diversification scores for all
    50 stocks. period: 1y | 2y | 3y | 5y
    """
    try:
        from correlation_matrix import build_returns_matrix, PERIOD_DAYS
        period_days = PERIOD_DAYS.get(period, 504)
        returns, sector_map, short_map = build_returns_matrix(period_days)
        if returns.empty:
            raise HTTPException(status_code=503, detail="No price data available")

        corr = returns.corr()

        # Diversification score: avg |correlation| with all other stocks
        import numpy as np # pyright: ignore[reportMissingImports]
        abs_corr = corr.abs()
        np.fill_diagonal(abs_corr.values, float('nan'))
        div_scores = abs_corr.mean(axis=1, skipna=True)

        # Top 10 most correlated pairs
        upper = corr.where(
            np.triu(np.ones(corr.shape), k=1).astype(bool)
        )
        top_pairs = (
            upper.stack()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
            .rename(columns={'level_0': 'ticker_a', 'level_1': 'ticker_b', 0: 'correlation'})
            .to_dict(orient='records')
        )

        return {
            "period"          : period,
            "n_stocks"        : len(corr),
            "avg_correlation" : round(float(upper.stack().mean()), 4),
            "diversification" : {
                t: round(float(v), 4)
                for t, v in div_scores.sort_values().items()
            },
            "top_correlated_pairs": [
                {
                    "ticker_a"   : r["ticker_a"],
                    "ticker_b"   : r["ticker_b"],
                    "correlation": round(r["correlation"], 4),
                    "sector_a"   : sector_map.get(r["ticker_a"], ""),
                    "sector_b"   : sector_map.get(r["ticker_b"], ""),
                }
                for r in top_pairs
            ],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# ═══════════════════════════════════════════════════════════
#  /tail-risk routes
# ═══════════════════════════════════════════════════════════


@app.get("/tail-risk")
async def get_tail_risk(current_user=Depends(get_current_user_optional)):
    """
    Returns the current Tail Risk Index and all component scores.
    Runs a live scan across all 50 stocks.
    """
    try:
        from src.tail_risk import TailRiskMonitor
        monitor = TailRiskMonitor()
        report  = monitor.scan_from_db(period_days=252)

        return {
            "tri"               : report.tri,
            "level"             : report.level,
            "halt_trading"      : report.halt_trading,
            "components": {
                "volatility"    : report.vol_score,
                "kurtosis"      : report.kurtosis_score,
                "skew"          : report.skew_score,
                "correlation"   : report.correlation_score,
                "liquidity"     : report.liquidity_score,
                "var_breach"    : report.var_breach_score,
            },
            "supporting_data": {
                "realized_vol_ann"   : report.realized_vol_ann,
                "avg_kurtosis"       : report.avg_kurtosis,
                "avg_skew"           : report.avg_skew,
                "avg_pairwise_corr"  : report.avg_pairwise_corr,
                "var_breach_count"   : report.var_breach_count,
                "n_stocks"           : report.n_stocks,
            },
            "reasons"        : report.reasons,
            "recommendation" : report.recommendation,
            "timestamp"      : report.timestamp,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# ═══════════════════════════════════════════════════════════
#  /risk-parity routes
# ═══════════════════════════════════════════════════════════
@app.get("/risk-parity")
async def get_risk_parity(capital: float = 100000,
                           current_user=Depends(get_current_user_optional)):
    """
    Returns risk parity allocation for today's BUY signals.
    capital: portfolio size in ₹ (default 100000)
    """
    try:
        from src.risk_parity import RiskParityAllocator
        from src.paper_trader import fetch_latest_data
        from src.ensemble_model import get_ensemble_confidence
        from src.risk import RiskManager

        rm      = RiskManager(initial_capital=capital)
        rm.capital = capital
        signals = []
        close_map = {}

        for ticker in STOCK_UNIVERSE:
            try:
                df = fetch_latest_data(ticker)
                if df is None or len(df) < 30:
                    continue
                ens_conf, _, _ = get_ensemble_confidence(ticker, df)
                if ens_conf is None or ens_conf < 0.58:
                    continue
                price = float(df.iloc[-1]['Close'])
                shares, stop_loss, _ = rm.calculate_position_size(
                    capital=capital, price=price, confidence=ens_conf
                )
                if shares > 0:
                    signals.append({
                        'ticker': ticker, 'price': price,
                        'confidence': round(ens_conf, 4),
                        'shares': shares,
                        'stop_loss': round(stop_loss, 2),
                        'trade_value': round(shares * price, 2),
                    })
                    close_map[ticker] = df['Close']
            except Exception:
                pass

        allocator = RiskParityAllocator()
        adj_signals, result = allocator.allocate(signals, capital, close_map)

        return {
            "capital"        : capital,
            "n_positions"    : result.n_stocks,
            "total_deployed" : result.total_deployed,
            "cash_remaining" : result.cash_remaining,
            "target_risk_per_position": result.target_risk_per_pos,
            "correlation_penalty_applied": result.correlation_used,
            "positions": [
                {
                    "ticker"        : s['ticker'],
                    "price"         : s['price'],
                    "shares"        : s['shares'],
                    "trade_value"   : s['trade_value'],
                    "weight_pct"    : round(s['rp_weight'] * 100, 2),
                    "daily_vol_pct" : s['rp_daily_vol'],
                    "daily_risk_inr": result.actual_risk_contribs.get(s['ticker'], 0),
                    "stop_loss"     : s['stop_loss'],
                    "confidence"    : s['confidence'],
                }
                for s in adj_signals
            ],
            "updated_at": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recovery-status")
async def get_recovery_status(current_user=Depends(get_current_user_optional)):
    """Returns the current drawdown recovery state."""
    try:
        from src.drawdown_recovery import DrawdownRecoveryManager
        drm = DrawdownRecoveryManager()
        drm.load_state()
        return drm.get_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/scheduler/status")
def get_scheduler_status():
    """Returns the last 20 scheduler run records."""
    if not os.path.exists(DB_PATH):
        return {"logs": [], "message": "Database not initialised yet"}
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT job, status, message, ran_at FROM scheduler_log "
        "ORDER BY ran_at DESC LIMIT 20"
    ).fetchall()
    conn.close()
    return {
        "logs": [{"job": r[0], "status": r[1], "message": r[2], "ran_at": r[3]}
                 for r in rows]
    }

# ═══════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════

class RegisterBody(BaseModel):
    username: str
    email: str
    password: str

class LoginBody(BaseModel):
    username: str   # accepts username OR email
    password: str

@app.post("/auth/register")
def register(body: RegisterBody):
    """Register a new user account."""
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if len(body.username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    from src.auth import register_user
    ok, result = register_user(body.username, body.email, body.password)
    if not ok:
        raise HTTPException(400, result)
    return {"message": f"Account created! Welcome, {body.username}.", "user_id": result}

@app.post("/auth/login")
def login(body: LoginBody):
    """Login and receive a session token (valid 7 days)."""
    from src.auth import login_user
    ok, result, user = login_user(body.username, body.password)
    if not ok:
        raise HTTPException(401, result)
    return {"token": result, "user": user}

@app.get("/auth/me")
def me(user_id: int = Depends(get_current_user)):
    """Returns the logged-in user's profile."""
    from src.auth import get_user_by_id
    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(404, "User not found")
    return user

@app.post("/auth/logout")
def logout(authorization: str = Header(default="")):
    """Revokes the current session token."""
    token = authorization.replace("Bearer ", "").strip()
    if token:
        from src.auth import revoke_token
        revoke_token(token)
    return {"message": "Logged out"}

# ═══════════════════════════════════════════════════════════
#  WATCHLIST ROUTES (auth required)
# ═══════════════════════════════════════════════════════════

@app.get("/watchlist")
def get_watchlist(user_id: int = Depends(get_current_user)):
    """Returns the logged-in user's personal watchlist with live signals."""
    from src.auth import get_watchlist
    items = get_watchlist(user_id)
    result = []
    for item in items:
        ticker = item["ticker"]
        sig = get_signal_for(ticker)
        if sig:
            sig["added_at"] = item["added_at"]
            result.append(sig)
        else:
            result.append({"ticker": ticker, "added_at": item["added_at"], "error": "No signal"})
    return {"watchlist": result, "count": len(result)}

@app.post("/watchlist/{ticker}")
def add_ticker(ticker: str, user_id: int = Depends(get_current_user)):
    """Adds a stock to the user's watchlist."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    if t not in STOCK_UNIVERSE:
        raise HTTPException(400, f"{t} not in Nifty 50 universe")
    from src.auth import add_to_watchlist
    add_to_watchlist(user_id, t)
    return {"message": f"{t} added to your watchlist"}

@app.delete("/watchlist/{ticker}")
def remove_ticker(ticker: str, user_id: int = Depends(get_current_user)):
    """Removes a stock from the user's watchlist."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    from src.auth import remove_from_watchlist
    remove_from_watchlist(user_id, t)
    return {"message": f"{t} removed from your watchlist"}

# ═══════════════════════════════════════════════════════════
#  USER PORTFOLIO ROUTES (auth required)
# ═══════════════════════════════════════════════════════════

@app.get("/user/portfolio")
def get_user_portfolio(user_id: int = Depends(get_current_user)):
    """Returns the logged-in user's personal paper trades."""
    from src.auth import get_user_trades
    trades = get_user_trades(user_id)
    closed = [t for t in trades if t["status"] == "CLOSED"]
    open_t = [t for t in trades if t["status"] == "OPEN"]
    wins   = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
    total_pnl = sum((t.get("pnl") or 0) for t in closed)
    return {
        "trades"        : trades,
        "open_count"    : len(open_t),
        "closed_count"  : len(closed),
        "wins"          : wins,
        "losses"        : len(closed) - wins,
        "total_pnl"     : round(total_pnl, 2),
        "win_rate"      : round(wins / len(closed) * 100, 1) if closed else 0
    }

class TradeBody(BaseModel):
    ticker: str
    shares: int
    price: float
    stop_loss: float = 0
    confidence: float = 0

@app.post("/user/portfolio/trade")
def add_trade(body: TradeBody, user_id: int = Depends(get_current_user)):
    """Manually adds a paper trade to the user's portfolio."""
    t = body.ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    from src.auth import add_user_trade
    add_user_trade(user_id, {
        "ticker": t, "shares": body.shares, "price": body.price,
        "stop_loss": body.stop_loss, "trade_value": body.shares * body.price,
        "date": str(date.today()), "confidence": body.confidence
    })
    return {"message": f"Trade recorded: BUY {body.shares} × {t} @ ₹{body.price}"}

# ═══════════════════════════════════════════════════════════
#  ADMIN ROUTES (admin users only)
# ═══════════════════════════════════════════════════════════

def require_admin(user_id: int = Depends(get_current_user)):
    from src.auth import get_user_by_id
    user = get_user_by_id(user_id)
    if not user or not user.get("is_admin"):
        raise HTTPException(403, "Admin access required")
    return user_id

@app.get("/admin/users")
def list_users(user_id: int = Depends(require_admin)):
    """Returns all registered users (admin only)."""
    from src.auth import get_all_users
    users = get_all_users()
    return {"users": users, "count": len(users)}



#--------------------------------------------------

"""
api_options_routes.py — QuantAI Options API Routes
===================================================
Paste these routes into the bottom of src/api.py and add the imports below.

ADD TO TOP OF src/api.py:
─────────────────────────
from src.options_pricing import (
    bsm_price, bsm_greeks, implied_volatility,
    get_lot_size, RISK_FREE_RATE,
)
from src.options_chain import fetch_full_chain, build_iv_surface
from src.options_strategies import (
    suggest_strategy, long_call, long_put,
    bull_call_spread, bear_put_spread,
    long_straddle, iron_condor,
)

ROUTES ADDED:
─────────────
  POST /options/price          — BSM price + all Greeks for one option
  POST /options/iv             — Solve implied volatility from market price
  GET  /options/chain/{ticker} — Full enriched chain with Greeks
  POST /options/strategy       — Build any named strategy with payoff curve
  GET  /options/suggest/{ticker} — AI strategy recommendation from live signal
  GET  /options/iv-surface/{ticker} — IV surface across all expiries
  GET  /options/scan           — Scan all BUY/SELL signals → strategy recommendations
"""

from pydantic import BaseModel # type: ignore
from typing import Optional, List
from datetime import datetime


# ── Pydantic models ───────────────────────────────────────

class OptionPriceRequest(BaseModel):
    ticker      : str
    spot        : float
    strike      : float
    dte         : int              # days to expiry
    sigma       : float            # annualised vol e.g. 0.25
    option_type : str = 'call'     # 'call' or 'put'
    r           : float = RISK_FREE_RATE


class IVRequest(BaseModel):
    market_price: float
    spot        : float
    strike      : float
    dte         : int
    option_type : str   = 'call'
    r           : float = RISK_FREE_RATE


class StrategyRequest(BaseModel):
    strategy    : str              # 'long_call' | 'long_put' | 'bull_call_spread' |
                                   # 'bear_put_spread' | 'long_straddle' | 'iron_condor'
    ticker      : str
    spot        : float
    dte         : int
    sigma       : float
    K1          : Optional[float] = None   # primary / lower strike
    K2          : Optional[float] = None   # secondary / upper strike
    K3          : Optional[float] = None   # condor: call sell
    K4          : Optional[float] = None   # condor: call buy
    r           : float = RISK_FREE_RATE


# ══════════════════════════════════════════════════════════
#  ROUTE 1 — BSM Price + Greeks
# ══════════════════════════════════════════════════════════

@app.post("/options/price")
def price_option(body: OptionPriceRequest):
    """
    Price a single European option and return all Greeks.

    Example request:
    {
      "ticker": "RELIANCE",
      "spot": 2850,
      "strike": 2900,
      "dte": 28,
      "sigma": 0.24,
      "option_type": "call"
    }
    """
    ticker = body.ticker.upper()
    if not ticker.endswith('.NS'):
        ticker += '.NS'

    T        = body.dte / 365
    lot_size = get_lot_size(ticker)

    try:
        price  = bsm_price(body.spot, body.strike, T, body.r, body.sigma, body.option_type)
        greeks = bsm_greeks(body.spot, body.strike, T, body.r, body.sigma, body.option_type)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {
        'ticker'         : ticker,
        'spot'           : body.spot,
        'strike'         : body.strike,
        'dte'            : body.dte,
        'option_type'    : body.option_type,
        'sigma_pct'      : round(body.sigma * 100, 2),
        'price_per_share': round(price, 2),
        'price_per_lot'  : round(price * lot_size, 2),
        'lot_size'       : lot_size,
        'greeks'         : greeks,
        'greeks_per_lot' : {
            k: round(v * lot_size, 4) for k, v in greeks.items()
        },
        'priced_at'      : datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 2 — Implied Volatility Solver
# ══════════════════════════════════════════════════════════

@app.post("/options/iv")
def solve_iv(body: IVRequest):
    """
    Back-solve Black-Scholes for implied volatility given a market price.

    Example:
    { "market_price": 62.50, "spot": 2850, "strike": 2900, "dte": 28 }
    """
    T  = body.dte / 365
    iv = implied_volatility(
        body.market_price, body.spot, body.strike,
        T, body.r, body.option_type
    )
    if iv is None:
        raise HTTPException(400,
            "No valid IV found. Check that market_price > intrinsic value.")

    greeks = bsm_greeks(body.spot, body.strike, T, body.r, iv, body.option_type)
    return {
        'implied_volatility_pct': round(iv * 100, 2),
        'implied_volatility'    : round(iv, 6),
        'greeks'                : greeks,
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 3 — Live Options Chain
# ══════════════════════════════════════════════════════════

@app.get("/options/chain/{ticker}")
def get_options_chain(
    ticker     : str,
    expiry_idx : int   = 0,
    n_strikes  : int   = 10,
):
    """
    Fetch live NSE options chain enriched with BSM Greeks.

    Path param:  ticker     e.g. RELIANCE or NIFTY
    Query params:
        expiry_idx  0 = nearest expiry (default), 1 = next, etc.
        n_strikes   strikes to return each side of ATM (default 10)

    Returns: spot, expiry, DTE, PCR, max_pain, IV rank,
             calls + puts DataFrames with Delta/Gamma/Theta/Vega per row.
    """
    t = ticker.upper()
    if not t.endswith('.NS') and t not in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
        t += '.NS'

    chain_data = fetch_full_chain(t, expiry_idx=expiry_idx)
    if chain_data is None:
        raise HTTPException(404, f"Could not fetch options chain for {t}. "
                                  "Check ticker or try again after market open.")

    spot  = chain_data['spot']
    calls = chain_data['calls']
    puts  = chain_data['puts']

    # Filter to n_strikes closest to ATM on each side
    def nearest_n(df, n):
        if df.empty:
            return []
        idx = df.iloc[(df['strike'] - spot).abs().argsort()[:n]].sort_values('strike')
        return idx.to_dict(orient='records')

    return {
        'ticker'   : t,
        'spot'     : spot,
        'expiry'   : chain_data['expiry'],
        'dte'      : chain_data['dte'],
        'atm_iv'   : chain_data['atm_iv'],
        'hv_30'    : chain_data['hv_30'],
        'iv_rank'  : chain_data['iv_rank'],
        'pcr_oi'   : chain_data['pcr_oi'],
        'pcr_vol'  : chain_data['pcr_vol'],
        'max_pain' : chain_data['max_pain'],
        'lot_size' : get_lot_size(t),
        'calls'    : nearest_n(calls, n_strikes),
        'puts'     : nearest_n(puts,  n_strikes),
        'fetched_at': datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 4 — Build a Named Strategy
# ══════════════════════════════════════════════════════════

@app.post("/options/strategy")
def build_strategy(body: StrategyRequest):
    """
    Build any supported strategy and return full P&L mechanics + payoff curve.

    Strategies:
      long_call, long_put, bull_call_spread, bear_put_spread,
      long_straddle, iron_condor

    Payoff curve: 200-point list of (spot, pnl) pairs for charting.

    Example (bull call spread on RELIANCE):
    {
      "strategy": "bull_call_spread",
      "ticker": "RELIANCE",
      "spot": 2850, "dte": 28, "sigma": 0.24,
      "K1": 2850, "K2": 2950
    }
    """
    ticker   = body.ticker.upper()
    if not ticker.endswith('.NS'):
        ticker += '.NS'

    T        = body.dte / 365
    lot_size = get_lot_size(ticker)
    r        = body.r
    sigma    = body.sigma
    spot     = body.spot
    name     = body.strategy.lower()

    K1, K2, K3, K4 = body.K1, body.K2, body.K3, body.K4

    # Default strikes (ATM / ±3% / ±6%) when not supplied
    step = 100 if spot > 5000 else 50
    atm  = round(spot / step) * step
    if K1 is None: K1 = atm
    if K2 is None: K2 = round((spot * 1.06) / step) * step
    if K3 is None: K3 = round((spot * 1.03) / step) * step
    if K4 is None: K4 = round((spot * 1.06) / step) * step

    try:
        if name == 'long_call':
            result = long_call(spot, K1, T, sigma, r, lot_size)
        elif name == 'long_put':
            result = long_put(spot, K1, T, sigma, r, lot_size)
        elif name == 'bull_call_spread':
            result = bull_call_spread(spot, K1, K2, T, sigma, r, lot_size)
        elif name == 'bear_put_spread':
            result = bear_put_spread(spot, K2, K1, T, sigma, r, lot_size)
        elif name == 'long_straddle':
            result = long_straddle(spot, T, sigma, r, lot_size)
        elif name == 'iron_condor':
            K_pb = round((spot * 0.94) / step) * step
            K_ps = round((spot * 0.97) / step) * step
            K_cs = round((spot * 1.03) / step) * step
            K_cb = round((spot * 1.06) / step) * step
            result = iron_condor(spot, K_pb, K_ps, K_cs, K_cb, T, sigma, r, lot_size)
        else:
            raise HTTPException(400, f"Unknown strategy '{body.strategy}'. "
                "Valid: long_call, long_put, bull_call_spread, bear_put_spread, "
                "long_straddle, iron_condor")
    except Exception as exc:
        raise HTTPException(500, f"Strategy construction failed: {exc}")

    mp = round(result.max_profit * lot_size, 2) if result.max_profit is not None else None
    ml = round(result.max_loss   * lot_size, 2) if result.max_loss   is not None else None

    return {
        'strategy'       : result.name,
        'ticker'         : ticker,
        'lot_size'       : lot_size,
        'net_premium'    : round(result.net_premium * lot_size, 2),
        'credit_or_debit': 'CREDIT' if result.net_premium >= 0 else 'DEBIT',
        'max_profit'     : mp,
        'max_loss'       : ml,
        'breakevens'     : result.breakevens,
        'net_greeks'     : result.net_greeks,
        'net_greeks_per_lot': {
            k: round(v * lot_size, 4) for k, v in result.net_greeks.items()
        },
        'legs'           : [
            {k: v for k, v in leg.items() if k not in ('sigma',)}
            for leg in result.legs
        ],
        'payoff_curve'   : {
            'spots'  : result.payoff_spots,
            'pnl'    : result.payoff_values,
        },
        'rationale'      : result.rationale,
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 5 — AI Strategy Suggestion (live signal + chain)
# ══════════════════════════════════════════════════════════

@app.get("/options/suggest/{ticker}")
def suggest_options_strategy(
    ticker     : str,
    expiry_idx : int   = 0,
    signal     : str   = None,   # override signal; None = auto from ensemble
    confidence : float = None,   # override confidence
):
    """
    Combine the QuantAI ensemble signal with a live options chain
    and return an AI-recommended strategy with full mechanics.

    Workflow:
      1. Run ensemble model for ticker → signal + confidence
      2. Fetch live chain → ATM IV, IV Rank, DTE
      3. Feed into suggest_strategy() → recommended strategy
      4. Build strategy with delta-appropriate strikes from live chain

    Returns complete strategy JSON identical to /options/strategy,
    plus the signal reasoning that drove the recommendation.
    """
    t = ticker.upper()
    if not t.endswith('.NS') and t not in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
        t += '.NS'

    # ── Step 1: Get ensemble signal ───────────────────────
    if signal and confidence:
        sig  = signal.upper()
        conf = float(confidence)
    else:
        raw = get_signal_for(t)   # existing helper from api.py
        if raw is None:
            raise HTTPException(404, f"Could not generate signal for {t}")
        sig  = raw.get('signal', 'HOLD')
        conf = raw.get('confidence', 0.0)

    # ── Step 2: Fetch live chain ──────────────────────────
    chain_data = fetch_full_chain(t, expiry_idx=expiry_idx)
    if chain_data is None:
        raise HTTPException(404, f"Could not fetch options chain for {t}")

    spot     = chain_data['spot']
    T        = chain_data['T']
    atm_iv   = chain_data['atm_iv'] / 100
    ivr      = chain_data['iv_rank']
    lot_size = get_lot_size(t)
    sigma    = atm_iv if atm_iv > 0.05 else 0.25

    # ── Step 3: Suggest strategy ──────────────────────────
    result, rationale = suggest_strategy(
        signal=sig, confidence=conf, iv_rank=ivr,
        spot=spot, T=T, sigma=sigma, lot_size=lot_size,
    )

    if result is None:
        return {
            'ticker'    : t,
            'signal'    : sig,
            'confidence': conf,
            'iv_rank'   : ivr,
            'strategy'  : None,
            'rationale' : rationale,
        }

    # ── Step 4: Enrich with live strike data ──────────────
    calls     = chain_data['calls']
    puts      = chain_data['puts']
    from src.options_strategies import select_strikes_from_chain
    live_strikes = select_strikes_from_chain(calls, puts, spot, sig)

    mp = round(result.max_profit * lot_size, 2) if result.max_profit is not None else None
    ml = round(result.max_loss   * lot_size, 2) if result.max_loss   is not None else None

    return {
        'ticker'          : t,
        'signal'          : sig,
        'confidence'      : round(conf, 4),
        'spot'            : spot,
        'expiry'          : chain_data['expiry'],
        'dte'             : chain_data['dte'],
        'atm_iv_pct'      : round(atm_iv * 100, 2),
        'iv_rank'         : ivr,
        'hv_30_pct'       : chain_data['hv_30'],
        'pcr_oi'          : chain_data['pcr_oi'],
        'max_pain'        : chain_data['max_pain'],
        'lot_size'        : lot_size,
        'strategy'        : result.name,
        'rationale'       : rationale,
        'net_premium'     : round(result.net_premium * lot_size, 2),
        'credit_or_debit' : 'CREDIT' if result.net_premium >= 0 else 'DEBIT',
        'max_profit'      : mp,
        'max_loss'        : ml,
        'breakevens'      : result.breakevens,
        'net_greeks'      : result.net_greeks,
        'live_strikes'    : live_strikes,
        'legs'            : [
            {k: v for k, v in leg.items() if k != 'sigma'}
            for leg in result.legs
        ],
        'payoff_curve'    : {
            'spots': result.payoff_spots,
            'pnl'  : result.payoff_values,
        },
        'suggested_at'    : datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 6 — IV Surface (for dashboard charting)
# ══════════════════════════════════════════════════════════

@app.get("/options/iv-surface/{ticker}")
def get_iv_surface(ticker: str, max_expiries: int = 4):
    """
    Return IV surface across strikes and expiries for 3D charting.

    Response shape: [{expiry, dte, strike, moneyness_pct, iv_pct}, ...]
    Suitable for Chart.js 3D scatter or heatmap on the dashboard.
    """
    t = ticker.upper()
    if not t.endswith('.NS') and t not in ('NIFTY', 'BANKNIFTY', 'FINNIFTY'):
        t += '.NS'

    surface = build_iv_surface(t, max_expiries=max_expiries)
    if surface.empty:
        raise HTTPException(404, f"Could not build IV surface for {t}")

    return {
        'ticker'  : t,
        'surface' : surface.to_dict(orient='records'),
        'expiries': surface['expiry'].unique().tolist(),
        'n_points': len(surface),
    }


# ══════════════════════════════════════════════════════════
#  ROUTE 7 — Full options scan across all BUY/SELL signals
# ══════════════════════════════════════════════════════════

@app.get("/options/scan")
def scan_options(
    min_confidence : float = 0.58,
    expiry_idx     : int   = 0,
    user_id        : int   = Depends(get_current_user_optional),
):
    """
    Scan all Nifty 50 stocks for options opportunities.
    Runs the ensemble model, picks BUY/SELL candidates,
    fetches chains, and returns AI strategy recommendations.

    Heavy endpoint (~30-60s for full scan).
    Query param min_confidence (default 0.58) controls strictness.
    """
    from src.data_collector import STOCK_UNIVERSE

    results  = []
    errors   = []
    tickers  = list(STOCK_UNIVERSE.keys())

    for ticker in tickers:
        try:
            raw = get_signal_for(ticker)
            if not raw:
                continue
            sig  = raw.get('signal', 'HOLD')
            conf = raw.get('confidence', 0.0)
            if conf < min_confidence or sig == 'HOLD':
                continue

            chain_data = fetch_full_chain(ticker, expiry_idx=expiry_idx)
            if not chain_data:
                continue

            spot     = chain_data['spot']
            T        = chain_data['T']
            atm_iv   = chain_data['atm_iv'] / 100
            ivr      = chain_data['iv_rank']
            lot_size = get_lot_size(ticker)
            sigma    = atm_iv if atm_iv > 0.05 else 0.25

            result, rationale = suggest_strategy(
                signal=sig, confidence=conf, iv_rank=ivr,
                spot=spot, T=T, sigma=sigma, lot_size=lot_size,
            )

            mp = round(result.max_profit * lot_size, 2) if result and result.max_profit else None
            ml = round(result.max_loss   * lot_size, 2) if result and result.max_loss   else None

            results.append({
                'ticker'         : ticker,
                'signal'         : sig,
                'confidence'     : round(conf, 4),
                'spot'           : spot,
                'expiry'         : chain_data['expiry'],
                'dte'            : chain_data['dte'],
                'atm_iv_pct'     : round(atm_iv * 100, 2),
                'iv_rank'        : ivr,
                'strategy'       : result.name if result else None,
                'net_premium'    : round(result.net_premium * lot_size, 2) if result else None,
                'max_profit'     : mp,
                'max_loss'       : ml,
                'breakevens'     : result.breakevens if result else [],
                'rationale'      : rationale,
            })
        except Exception as exc:
            errors.append({'ticker': ticker, 'error': str(exc)})

    results.sort(key=lambda x: -x['confidence'])
    return {
        'scanned'    : len(tickers),
        'candidates' : len(results),
        'results'    : results,
        'errors'     : errors,
        'scanned_at' : datetime.now().isoformat(),
    }

@app.get('/performance')
def get_performance():
    """Returns full performance metrics."""
    try:
        from src.performance_tracker import (
            load_all_trades, close_open_trades,
            calculate_metrics, sector_breakdown,
            confidence_breakdown, monthly_breakdown
        )
        trades  = load_all_trades()
        trades  = close_open_trades(trades)
        metrics = calculate_metrics(trades)
        return {
            'metrics'   : metrics,
            'by_sector' : sector_breakdown(trades),
            'by_confidence': confidence_breakdown(trades),
            'by_month'  : monthly_breakdown(trades),
        }
    except Exception as e:
        return {'error': str(e)}