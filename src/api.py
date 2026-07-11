"""
src/api.py — QuantAI FastAPI backend v4.0
New in v4: Multi-user auth, watchlist, per-user portfolio, admin panel, scheduler status.
"""
import os, time, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fastapi import FastAPI, HTTPException, Header, Depends, Request# type: ignore
from fastapi.middleware.cors import CORSMiddleware# type: ignore
from fastapi.staticfiles import StaticFiles# type: ignore
from pydantic import BaseModel# type: ignore
from datetime import datetime, date# type: ignore
import sqlite3, joblib, json, pandas as pd# type: ignore
from src.features import get_feature_dataset, add_features
from src.model import FEATURE_COLS
from src.data_collector import STOCK_UNIVERSE
import yfinance as yf# type: ignore

# ── In-memory cache store ───────────────────────────────
_CACHE: dict = {}

def _safe_num(val, digits=4, default=0.0):
    """
    Rounds a value to `digits` decimal places, but — critically — also
    catches NaN/Infinity, which round(float(x), n) passes through
    completely unchanged (no exception raised). Starlette's JSONResponse
    rejects NaN/Infinity at serialization time with "Out of range float
    values are not JSON compliant", which crashes the ENTIRE response
    (e.g. all 50 stocks in /signals) even if only one ticker's indicator
    was NaN (e.g. RSI/MACD/BB_Width during a warm-up period, or a stock
    with a data gap). Use this instead of a bare round(float(x), n)
    anywhere a computed number goes into a JSON response.
    """
    try:
        f = float(val)
        if not math.isfinite(f):
            return default
        return round(f, digits)
    except Exception:
        return default

def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and (time.time() - entry["at"]) < entry["ttl"]:
        return entry["data"]
    return None

def _cache_set(key: str, data, ttl: int = 600):
    _CACHE[key] = {"data": data, "at": time.time(), "ttl": ttl}

def _cache_clear(key: str = None):
    if key:
        _CACHE.pop(key, None)
    else:
        _CACHE.clear()


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

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "quantai.db")
# ^ file-relative, not CWD-relative — matches src/auth.py and src/database.py.
# A CWD-relative path here meant /health could report db_ok=True/False based
# on whatever folder you happened to launch uvicorn from, out of sync with
# where auth.py and database.py actually read/write the DB.

# ── Startup: init DB tables + warm cache in background ────
@app.on_event("startup")
def on_startup():
    """Init DB and kick off background cache warming."""
    try:
        from src.database import create_tables
        create_tables()
    except Exception:
        pass

    # Warm signals cache in background so first user request is fast
    import threading
    def _warm():
        try:
            time.sleep(3)   # let server fully start first
            get_all_signals(force=True)
        except Exception:
            pass
    threading.Thread(target=_warm, daemon=True).start()

    # Warm tail-risk cache too
    def _warm_tri():
        try:
            time.sleep(8)
            get_tail_risk()
        except Exception:
            pass
    threading.Thread(target=_warm_tri, daemon=True).start()

@app.get("/api/status")
def root():
    # Was @app.get("/") — moved here since "/" now serves the dashboard
    # itself (see the StaticFiles mount at the bottom of this file), so
    # visiting your deployed URL opens the dashboard directly instead of
    # a JSON status blob. Nothing in the frontend called GET "/" for data,
    # so this move is safe.
    cached = _cache_get("signals_all")
    return {
        "status"     : "QuantAI API running",
        "version"    : "4.0",
        "cache_hot"  : cached is not None,
        "endpoints"  : ["/signals", "/signal/{ticker}", "/tail-risk",
                        "/regime", "/fii-dii", "/news/{ticker}",
                        "/prices/{ticker}", "/portfolio",
                        "/paper-trade/positions", "/health"],
    }

@app.get("/health")
def health():
    """Lightweight health check — always fast, used by Docker HEALTHCHECK."""
    db_ok = os.path.exists(DB_PATH)
    return {
        "ok"         : True,
        "db"         : db_ok,
        "cache_keys" : list(_CACHE.keys()),
        "timestamp"  : datetime.now().isoformat(),
    }

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
    """Same as get_current_user but returns None instead of 401 — for public endpoints.

    This must NEVER raise. It runs as a FastAPI dependency, which means any
    exception here happens before the route handler's own try/except even
    starts — so a crash here produced an unhandled 500 that looked to the
    browser like the API was unreachable ("Can't reach the QuantAI API"),
    even though the route itself (e.g. /tail-risk) had perfectly good error
    handling inside its body. validate_token() is now hardened too, but we
    guard here as well since this dependency is shared by many routes.
    """
    try:
        token = authorization.replace("Bearer ", "").strip()
        if not token:
            return None
        from src.auth import validate_token
        return validate_token(token)
    except Exception:
        return None

# ── ML Signal helper ──────────────────────────────────────

def _classify_stock_regime(df):
    """
    Lightweight per-stock trend regime, independent of the market-wide
    detect_regime() in regime_detector.py (that one needs 200+ days of
    data across all 50 tickers and is meant for the overall market — it
    was never wired up per-stock). Every signal card/table/filter in the
    dashboard reads `s.regime` and `s.regime_score` per stock, so we
    compute a simple one here from the same 120-day df already in hand.

    Score is on a -1..+1 scale (matches the +/-0.15 thresholds already used
    to roll these up into the overall market_regime block in /signals).
    """
    try:
        close = df["Close"]
        if len(close) < 50:
            return "SIDEWAYS", 0.0
        current = float(close.iloc[-1])
        ma50    = float(close.rolling(50).mean().iloc[-1])
        trend   = 1.0 if current > ma50 else -1.0

        lookback = min(20, len(close) - 1)
        ret_20 = (current - float(close.iloc[-1 - lookback])) / float(close.iloc[-1 - lookback])
        momentum = max(-1.0, min(1.0, ret_20 / 0.05))  # +-5% => full scale

        score = round(0.6 * trend + 0.4 * momentum, 3)
        regime = "BULL" if score >= 0.15 else "BEAR" if score <= -0.15 else "SIDEWAYS"
        return regime, score
    except Exception:
        return "SIDEWAYS", 0.0


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

        regime, regime_score = _classify_stock_regime(df)

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
            "price"         : _safe_num(price, 2),
            "confidence"    : _safe_num(ens_conf, 4),
            "signal"        : signal,
            "regime"        : regime,
            "regime_score"  : _safe_num(regime_score, 3),
            "model_type"    : model_type,
            "models_used"   : models_used,
            "ml_count"      : ml_count,
            "rb_count"      : rb_count,
            "agreement"     : agreement,
            "individual"    : {k: _safe_num(v, 4) for k, v in (individual or {}).items()},
            "rule_based"    : rule_based_summary,
            "rsi"           : _safe_num(latest["RSI_14"], 2),
            "macd"          : _safe_num(latest["MACD"], 4),
            "bb_width"      : _safe_num(latest["BB_Width"], 4),
            "volume_ratio"  : _safe_num(latest["Volume_Ratio"], 2),
            "updated_at"    : datetime.now().isoformat(),
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

# ═══════════════════════════════════════════════════════════
#  PUBLIC ROUTES (no auth needed)
# ═══════════════════════════════════════════════════════════

@app.get("/signals")
def get_all_signals(force: bool = False, user_id: int = Depends(get_current_user_optional)):
    """
    Scans all 50 Nifty stocks in parallel (10 workers → ~5-8x faster).
    Results cached 10 min. Pass ?force=true to bypass cache.
    """
    CACHE_KEY = "signals_all"
    CACHE_TTL = 600

    if not force:
        cached = _cache_get(CACHE_KEY)
        if cached:
            cached["from_cache"] = True
            return cached

    # ── Parallel fetch ────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(get_signal_for, t): t for t in STOCK_UNIVERSE}
        for future in as_completed(futures):
            try:
                sig = future.result(timeout=30)
                if sig and not sig.get("error"):
                    results.append(sig)
            except Exception:
                pass

    buys        = [r for r in results if r.get("signal") == "BUY"]
    scores      = [r.get("regime_score", 0) for r in results if "regime_score" in r]
    avg_score   = round(sum(scores) / len(scores), 3) if scores else 0.0
    overall     = "BULL" if avg_score >= 0.15 else "BEAR" if avg_score <= -0.15 else "SIDEWAYS"

    payload = {
        "scan_time"    : datetime.now().isoformat(),
        "total_stocks" : len(results),
        "buy_signals"  : len(buys),
        "from_cache"   : False,
        "market_regime": {
            "overall_regime" : overall,
            "avg_score"      : avg_score,
            "bull_count"     : sum(1 for r in results if r.get("regime") == "BULL"),
            "sideways_count" : sum(1 for r in results if r.get("regime") == "SIDEWAYS"),
            "bear_count"     : sum(1 for r in results if r.get("regime") == "BEAR"),
            "threshold"      : {"BULL": 0.55, "SIDEWAYS": 0.58, "BEAR": 0.65}.get(overall, 0.58),
        },
        "results": sorted(results, key=lambda x: x.get("confidence", 0), reverse=True)
    }
    _cache_set(CACHE_KEY, payload, CACHE_TTL)
    return payload

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
        score, articles, meta = get_sentiment(t, max_articles=limit)
        modifier        = get_sentiment_modifier(score)
        return {
            "ticker"          : t,
            "sentiment_score" : score,
            "sentiment_label" : sentiment_label(score),
            "modifier"        : modifier,
            "articles"        : articles[:limit],
            "stale"           : meta.get("stale", False),
            "last_updated"    : meta.get("fetched_at"),
        }
    except Exception as e:
        return {"ticker": t, "error": str(e), "articles": []}

@app.get("/prices/{ticker}")
def get_prices(ticker: str, days: int = 90):
    """Returns historical OHLCV prices from database."""
    t = ticker.upper()
    if not t.endswith(".NS"):
        t += ".NS"
    try:
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume "
            "FROM prices WHERE ticker=? ORDER BY date DESC LIMIT ?",
            conn, params=(t, days)
        )
        conn.close()
    except Exception as e:
        # This route had no try/except at all before — any DB hiccup (file
        # missing, table missing, locked file) crashed the request with an
        # unhandled 500, which the browser reports as "can't reach the API"
        # even though every other part of the API was fine.
        return {"ticker": t, "days": 0, "prices": [], "error": str(e)}

    # A NULL in the DB (e.g. missing Open/Volume for a given day, common in
    # real market data) becomes NaN in pandas for numeric columns. Raw NaN
    # crashes Starlette's JSONResponse ("Out of range float values are not
    # JSON compliant") — this is the exact error seen when opening a stock's
    # price chart. Convert NaN -> None first; None serializes as JSON null,
    # which the frontend chart can skip/interpolate instead of crashing.
    df_clean = df.astype(object).where(df.notna(), None)
    return {"ticker": t, "days": len(df_clean), "prices": df_clean.to_dict(orient="records")}

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
        # .values can return a read-only array under pandas' Copy-on-Write
        # mode (default in recent pandas versions) — np.fill_diagonal writes
        # in-place and crashed with "assignment destination is read-only" /
        # "underlying array is read-only". .to_numpy(copy=True) guarantees a
        # writable array regardless of pandas version/CoW setting.
        abs_corr_np = abs_corr.to_numpy(copy=True)
        np.fill_diagonal(abs_corr_np, float('nan'))
        abs_corr = pd.DataFrame(abs_corr_np, index=abs_corr.index, columns=abs_corr.columns)
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
            "avg_correlation" : _safe_num(upper.stack().mean(), 4),
            "diversification" : {
                t: _safe_num(v, 4)
                for t, v in div_scores.sort_values().items()
            },
            "top_correlated_pairs": [
                {
                    "ticker_a"   : r["ticker_a"],
                    "ticker_b"   : r["ticker_b"],
                    "correlation": _safe_num(r["correlation"], 4),
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
def get_tail_risk(current_user=Depends(get_current_user_optional)):
    """
    Returns the Tail Risk Index.
    - Always returns HTTP 200 (errors in body, not as HTTP 5xx)
    - Results cached 30 minutes
    - Hard 25-second timeout to prevent frontend hang
    """
    CACHE_KEY = "tail_risk"
    cached = _cache_get(CACHE_KEY)
    if cached:
        cached["from_cache"] = True
        return cached

    import concurrent.futures as _cf

    def _compute():
        from src.tail_risk import TailRiskMonitor
        m = TailRiskMonitor()
        return m.scan_from_db(period_days=60)   # 60 days = fast

    report = None
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_compute)
        try:
            report = future.result(timeout=25)   # hard 25s cap
        except _cf.TimeoutError:
            return {
                "tri": None, "level": "TIMEOUT", "halt_trading": False,
                "components": {}, "supporting_data": {}, "reasons": [],
                "recommendation": "TRI scan timed out — it will be faster once cache warms up.",
                "timestamp": datetime.now().isoformat(),
                "error": "Computation timed out (>25 s). Will retry next call.",
                "from_cache": False,
            }
        except Exception as e:
            err = str(e)
            user_msg = (
                "Run python pipeline.py first to build the price database."
                if "not found" in err.lower() or "not enough" in err.lower()
                else err
            )
            return {
                "tri": None, "level": "ERROR", "halt_trading": False,
                "components": {}, "supporting_data": {}, "reasons": [],
                "recommendation": user_msg,
                "timestamp": datetime.now().isoformat(),
                "error": err,
                "from_cache": False,
            }

    payload = {
        "tri"          : _safe_num(report.tri, 1),
        "level"        : str(getattr(report, "level", "UNKNOWN")),
        "halt_trading" : bool(getattr(report, "halt_trading", False)),
        "components"   : {
            "volatility"  : _safe_num(getattr(report, "vol_score",         0), 1),
            "kurtosis"    : _safe_num(getattr(report, "kurtosis_score",    0), 1),
            "skew"        : _safe_num(getattr(report, "skew_score",        0), 1),
            "correlation" : _safe_num(getattr(report, "correlation_score", 0), 1),
            "liquidity"   : _safe_num(getattr(report, "liquidity_score",   0), 1),
            "var_breach"  : _safe_num(getattr(report, "var_breach_score",  0), 1),
        },
        "supporting_data": {
            "realized_vol_ann"  : _safe_num(getattr(report, "realized_vol_ann",   0), 4),
            "avg_kurtosis"      : _safe_num(getattr(report, "avg_kurtosis",       0), 3),
            "avg_skew"          : _safe_num(getattr(report, "avg_skew",           0), 3),
            "avg_pairwise_corr" : _safe_num(getattr(report, "avg_pairwise_corr",  0), 3),
            "var_breach_count"  : int(getattr(report, "var_breach_count", 0) or 0),
            "n_stocks"          : int(getattr(report, "n_stocks",         0) or 0),
        },
        "reasons"        : list(getattr(report, "reasons",        []) or []),
        "recommendation" : str(getattr(report, "recommendation",  "") or ""),
        "timestamp"      : str(getattr(report, "timestamp",       datetime.now().isoformat())),
        "error"          : None,
        "from_cache"     : False,
    }
    _cache_set(CACHE_KEY, payload, 1800)
    return payload


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
        # Greeks (delta/gamma/theta/vega) can be NaN/inf for degenerate
        # inputs (e.g. zero time-to-expiry, deep ITM/OTM) — same crash
        # class as the price-chart and tail-risk bugs. Sanitize before
        # to_dict() so a single bad strike doesn't 500 the whole chain.
        idx = idx.astype(object).where(idx.notna(), None)
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
        'theoretical': chain_data.get('theoretical', False),
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

    # Validate early with a clear message, instead of letting a bad input
    # (e.g. spot=0.3, a likely typo — real NSE stock prices are never
    # this small) reach the Black-Scholes math and crash with a bare
    # "float division by zero".
    if spot is None or spot <= 0:
        raise HTTPException(400, f"Spot price must be a positive number (got {spot}). "
                                  "Did you mean to enter a price in the hundreds/thousands?")
    if spot < 1:
        raise HTTPException(400, f"Spot price of {spot} looks too small to be a real stock "
                                  "price — check for a typo (e.g. did you mean to enter the IV "
                                  "or confidence value here instead?).")
    if body.dte is None or body.dte <= 0:
        raise HTTPException(400, f"Days to expiry must be positive (got {body.dte}).")
    if sigma is None or sigma <= 0:
        raise HTTPException(400, f"Implied vol must be positive (got {sigma}). "
                                  "Use a decimal like 0.24 for 24%, not a percentage like 24.")

    K1, K2, K3, K4 = body.K1, body.K2, body.K3, body.K4

    # Default strikes (ATM / ±3% / ±6%) when not supplied
    step = 100 if spot > 5000 else 50
    atm  = round(spot / step) * step
    if atm <= 0:
        # A valid (spot >= 1) but still small spot relative to `step` can
        # still round down to 0 (e.g. spot=20 with step=50) — fall back
        # to rounding to the nearest whole rupee instead of the coarser
        # 50/100 step, rather than ever handing a 0 strike to BSM pricing.
        atm = max(1, round(spot))
        step = max(1, round(spot * 0.03)) or 1
    if K1 is None: K1 = atm
    if K2 is None: K2 = round((spot * 1.06) / step) * step or atm
    if K3 is None: K3 = round((spot * 1.03) / step) * step or atm
    if K4 is None: K4 = round((spot * 1.06) / step) * step or atm

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
        'theoretical'     : chain_data.get('theoretical', False),
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

    # IV solving (Newton-Raphson / bisection on Black-Scholes) can fail to
    # converge and return NaN for illiquid/degenerate strikes — same crash
    # class as elsewhere in this file. Sanitize before to_dict().
    surface = surface.astype(object).where(surface.notna(), None)

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
                'theoretical'    : chain_data.get('theoretical', False),
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

@app.post("/signals/refresh")
def refresh_signals_cache():
    """Force-clears the signals cache. Next /signals call will re-scan."""
    _cache_clear("signals_all")
    return {"cleared": True, "message": "Cache cleared — next /signals call will scan live."}

# ── Paper Trade endpoints ──────────────────────────────────

class PaperTradeRequest(BaseModel):
    ticker: str
    shares: int
    price: Optional[float] = None   # None = use current live price
    # Was `price: float = None` — in Pydantic v2, a bare `float` type does
    # NOT automatically accept None just because the default is None (that
    # implicit widening was a Pydantic v1 behavior). Explicitly sending
    # price:null (exactly what the frontend does when the price field is
    # left blank for "use live market price") failed validation with a 422
    # every single time — every "Auto price" order was rejected before it
    # ever reached the order logic.

@app.post('/paper-trade/buy')
def paper_trade_buy(req: PaperTradeRequest, user=Depends(get_current_user)):
    """Open a paper BUY position."""
    import json as _json
    from src.paper_trader import fetch_latest_data
    ticker = req.ticker.upper()
    if not ticker.endswith(".NS"): ticker += ".NS"

    price = req.price
    if not price:
        try:
            df = fetch_latest_data(ticker, lookback_days=5)
            price = float(df.iloc[-1]["Close"]) if df is not None and not df.empty else None
        except Exception:
            price = None
    # "if not price" does NOT catch NaN — float('nan') is truthy in Python —
    # so a bad/missing price could silently become a NaN-priced trade record
    # that later crashes any endpoint returning it. Check explicitly.
    if price is None or not math.isfinite(price):
        return {"success": False, "error": "Could not get live price. Provide price manually."}

    trade = {
        "ticker": ticker, "shares": int(req.shares),
        "price": _safe_num(price, 2),
        "trade_value": _safe_num(price * int(req.shares), 2),
        "model_type": "Manual", "date": str(date.today()), "status": "OPEN",
    }
    from src.user_db import save_user_trade, get_user_portfolio, update_user_capital
    portfolio = get_user_portfolio(user)
    cost = trade["trade_value"]
    if cost > portfolio["capital"]:
        return {"success": False, "error": f"Insufficient capital. Available: ₹{portfolio['capital']:,.0f}"}
    save_user_trade(user, trade)
    update_user_capital(user, portfolio["capital"] - cost)
    return {"success": True, "trade": trade, "capital_remaining": portfolio["capital"] - cost}

@app.post('/paper-trade/sell')
def paper_trade_sell(req: PaperTradeRequest, user=Depends(get_current_user)):
    """Close a paper position and realise P&L."""
    from src.paper_trader import fetch_latest_data
    ticker = req.ticker.upper()
    if not ticker.endswith(".NS"): ticker += ".NS"

    price = req.price
    if not price:
        try:
            df = fetch_latest_data(ticker, lookback_days=5)
            price = float(df.iloc[-1]["Close"]) if df is not None and not df.empty else None
        except Exception: price = None
    if not price:
        return {"success": False, "error": "Could not get live price."}

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT id, price, shares FROM user_trades WHERE user_id=? AND ticker=? AND status='OPEN' ORDER BY date DESC LIMIT 1",
              (user, ticker))
    row = c.fetchone()
    if not row:
        conn.close()
        return {"success": False, "error": f"No open position found for {ticker}"}

    trade_id, entry_price, shares = row
    shares = min(int(req.shares), int(shares))
    pnl    = (float(price) - float(entry_price)) * shares
    proceeds = float(price) * shares

    c.execute("UPDATE user_trades SET status='CLOSED', pnl=? WHERE id=?", (round(pnl,2), trade_id))
    conn.commit()
    conn.close()

    from src.user_db import get_user_portfolio, update_user_capital
    portfolio = get_user_portfolio(user)
    update_user_capital(user, portfolio["capital"] + proceeds)

    return {"success": True, "pnl": round(pnl, 2), "proceeds": round(proceeds, 2), "exit_price": price}

@app.get('/paper-trade/positions')
def get_paper_positions(user=Depends(get_current_user)):
    """Returns open paper positions with live P&L."""
    from src.user_db import get_user_trades, get_user_portfolio
    trades    = get_user_trades(user, limit=50)
    open_pos  = [t for t in trades if t.get("status") == "OPEN"]
    portfolio = get_user_portfolio(user)
    return {"positions": open_pos, "capital": portfolio["capital"]}

@app.get('/paper-trade/history')
def get_paper_history(limit: int = 30, user=Depends(get_current_user)):
    """Returns closed paper trades."""
    from src.user_db import get_user_trades
    trades  = get_user_trades(user, limit=limit)
    closed  = [t for t in trades if t.get("status") == "CLOSED"]
    total_pnl = sum(t.get("pnl", 0) or 0 for t in closed)
    wins      = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
    return {"trades": closed, "total_pnl": round(total_pnl,2), "win_rate": round(wins/len(closed)*100,1) if closed else 0}

@app.post('/paper-trade/reset')
def reset_paper_portfolio(user=Depends(get_current_user)):
    """Resets paper portfolio to ₹1,00,000."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE user_trades SET status='CLOSED' WHERE user_id=? AND status='OPEN'", (user,))
    conn.execute("UPDATE user_portfolios SET capital=100000, peak=100000 WHERE user_id=?", (user,))
    conn.commit(); conn.close()
    return {"success": True, "capital": 100000}

@app.post('/telegram/connect-link')
def telegram_connect_link(user=Depends(get_current_user)):
    """
    Generates a one-time deep link that starts a chat with the QuantAI
    Telegram bot. Opening it and hitting Start in Telegram links that
    Telegram account to this QuantAI user — no manual chat_id copying.
    """
    bot_username = os.environ.get("TELEGRAM_BOT_USERNAME", "")
    if not bot_username:
        raise HTTPException(
            500,
            "TELEGRAM_BOT_USERNAME is not configured on the server. "
            "Set it to your bot's @username (without the @) in your .env / "
            "Render environment variables."
        )
    from src.user_db import create_telegram_link_token
    token = create_telegram_link_token(user)
    return {"deep_link": f"https://t.me/{bot_username}?start={token}"}


@app.get('/telegram/status')
def telegram_status(user=Depends(get_current_user)):
    """Whether this user currently has a Telegram chat connected."""
    from src.user_db import get_telegram_chat_id
    chat_id = get_telegram_chat_id(user)
    return {"connected": chat_id is not None}


@app.post('/telegram/disconnect')
def telegram_disconnect(user=Depends(get_current_user)):
    """Removes this user's Telegram connection — they'll stop receiving alerts."""
    from src.user_db import disconnect_telegram
    disconnect_telegram(user)
    return {"success": True}


@app.post('/telegram/webhook')
async def telegram_webhook(request: Request):
    """
    Telegram calls this URL directly (configure via setup_telegram_webhook.py
    after deploying) whenever someone messages the bot. We only care about
    "/start <token>" — everything else is ignored. This endpoint is public
    by necessity (Telegram, not your users, calls it), so it does the
    minimum possible: validate the token, attach the chat_id, done.
    """
    try:
        update = await request.json()
    except Exception:
        return {"ok": True}   # never error back to Telegram — it'll just retry

    message = update.get("message") or update.get("edited_message") or {}
    text    = (message.get("text") or "").strip()
    chat_id = (message.get("chat") or {}).get("id")

    from src.alerts import _send_to

    if not text.startswith("/start") or chat_id is None:
        return {"ok": True}

    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        _send_to(chat_id,
                 "👋 Welcome to QuantAI! Please use the \"Add Telegram Bot\" "
                 "button in your dashboard settings to connect your account "
                 "— this link expired or wasn't a valid connection link.")
        return {"ok": True}

    token = parts[1].strip()
    from src.user_db import resolve_telegram_link_token, set_telegram_chat_id
    from src.auth import get_user_by_id
    user_id = resolve_telegram_link_token(token)
    if user_id is None:
        _send_to(chat_id,
                 "⚠️ This connection link is invalid or has expired. "
                 "Generate a new one from your QuantAI dashboard settings.")
        return {"ok": True}

    set_telegram_chat_id(user_id, chat_id)
    user_info = get_user_by_id(user_id)
    username  = user_info["username"] if user_info else "there"
    _send_to(chat_id,
             f"✅ Connected! Hi {username}, you'll now receive QuantAI signal "
             f"alerts, risk warnings, and daily summaries here. You can "
             f"disconnect anytime from your dashboard settings.")
    return {"ok": True}


@app.get('/fii-dii')
def get_fii_dii(days: int = 10):
    """Returns recent FII/DII institutional flow data + signal."""
    try:
        from src.fii_dii import get_recent_data, get_flow_signal
        signal = get_flow_signal(days_avg=5)
        recent = get_recent_data(days=days)
        return {
            'signal'     : signal['signal'],
            'modifier'   : signal['modifier'],
            'fii_net_avg': signal['fii_net_avg'],
            'dii_net_avg': signal['dii_net_avg'],
            'combined_avg': signal['combined_avg'],
            'days_used'  : signal['days_used'],
            'data'       : recent,
            'note'       : signal.get('message', ''),
        }
    except Exception as e:
        return {'error': str(e), 'data': [], 'signal': 'NEUTRAL', 'modifier': 1.0}

@app.post('/fii-dii/refresh')
def refresh_fii_dii(user_id: int = Depends(get_current_user)):
    """Triggers a live fetch of FII/DII data from NSE (auth required)."""
    try:
        from src.fii_dii import fetch_and_store
        return fetch_and_store(days=30)
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get('/regime')
def get_market_regime():
    """
    Returns the overall market regime (BULL/BEAR/SIDEWAYS/VOLATILE), computed
    from all 50 stocks' price history via src/regime_detector.py.

    NOTE: this endpoint was referenced by the frontend (QuantAPI.regime())
    and listed in the root endpoint list, but was never actually implemented
    -- any call to it was returning a 404. Adding it here.
    """
    try:
        from src.regime_detector import detect_regime
        return detect_regime()
    except Exception as e:
        return {
            'regime': 'SIDEWAYS', 'composite': 0.5, 'scores': {}, 'details': {},
            'error': str(e),
            'note': 'Run python pipeline.py first to build the price database.'
                    if 'not found' in str(e).lower() or 'no such table' in str(e).lower() else str(e),
        }

@app.get('/regime/{ticker}')
def get_ticker_regime(ticker: str):
    """
    Per-stock trend regime for a single ticker (BULL/BEAR/SIDEWAYS), same
    classification used to populate the `regime` field on each /signals
    result. This was also referenced by the frontend (QuantAPI.regimeTicker)
    but never implemented on the backend.
    """
    try:
        df = yf.download(ticker, period="120d", progress=False, auto_adjust=True)
        if df.empty:
            return {"ticker": ticker, "regime": "SIDEWAYS", "regime_score": 0.0, "error": "No price data"}
        if hasattr(df.columns, 'levels'):
            df.columns = [col[0] for col in df.columns]
        regime, score = _classify_stock_regime(df)
        return {"ticker": ticker, "regime": regime, "regime_score": score}
    except Exception as e:
        return {"ticker": ticker, "regime": "SIDEWAYS", "regime_score": 0.0, "error": str(e)}

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


# ══════════════════════════════════════════════════════════
#  Serve the dashboard itself from this same service
# ══════════════════════════════════════════════════════════
# This MUST be the last thing registered. Starlette matches routes in the
# order they were added — every @app.get/@app.post above this already
# claimed its exact path, so this mount only ever catches requests that
# didn't match any of them (i.e. dashboard pages and static assets).
# Mounting it at "/" means your deployed URL opens the dashboard directly
# instead of a JSON status blob (that moved to /api/status above).
_dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount("/", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")