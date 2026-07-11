"""
src/fii_dii.py — FII/DII Institutional Flow Data
FII = Foreign Institutional Investors | DII = Domestic Institutional Investors

Fetches from NSE India live API and CSV archive, stores in SQLite,
returns a market-wide confidence modifier for the ensemble.

Signal logic (5-day average FII net, ₹ Crore):
  > +2000 → STRONGLY BULLISH → +8% confidence boost
  > +500  → BULLISH          → +4%
  -500 to 500 → NEUTRAL      →  0%
  < -500  → BEARISH          → -4%
  < -2000 → STRONGLY BEARISH → -8%
"""
import os, time, sqlite3, requests
import pandas as pd
from datetime import datetime, date, timedelta
from src.database import get_connection

# ── Table ─────────────────────────────────────────────────
def create_fii_dii_table():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fii_dii (
            date          TEXT PRIMARY KEY,
            fii_buy       REAL DEFAULT 0,
            fii_sell      REAL DEFAULT 0,
            fii_net       REAL DEFAULT 0,
            dii_buy       REAL DEFAULT 0,
            dii_sell      REAL DEFAULT 0,
            dii_net       REAL DEFAULT 0,
            combined_net  REAL DEFAULT 0,
            updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# ── NSE session ───────────────────────────────────────────
def _nse_session():
    s = requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    try:
        s.get('https://www.nseindia.com', timeout=10)
        time.sleep(0.8)
    except Exception:
        pass
    return s

# ── Fetch helpers ─────────────────────────────────────────
def _fetch_live():
    try:
        s   = _nse_session()
        r   = s.get('https://www.nseindia.com/api/fiidiiTradeReact',
                    headers={'Accept': 'application/json',
                             'Referer': 'https://www.nseindia.com/market-data/fii-dii-activity'},
                    timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list): return data
            if isinstance(data, dict) and 'data' in data: return data['data']
    except Exception:
        pass
    return []

def _fetch_csv(days=30):
    try:
        r = requests.get(
            'https://archives.nseindia.com/content/indices/ind_fiidii.csv',
            timeout=20, headers={'User-Agent': 'Mozilla/5.0'}
        )
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df.columns = [c.strip().lower().replace(' ','_') for c in df.columns]
            return df.head(days)
    except Exception:
        pass
    return pd.DataFrame()

def _f(v):
    try: return float(str(v).replace(',','').strip())
    except: return 0.0

def _parse_live(raw):
    fii = dii = None
    for item in raw:
        cat = str(item.get('category', item.get('name',''))).upper()
        if 'FII' in cat or 'FPI' in cat: fii = item
        elif 'DII' in cat: dii = item
    if not fii:
        return []
    fii_buy  = _f(fii.get('buyValue',  fii.get('buy_value',  0)))
    fii_sell = _f(fii.get('sellValue', fii.get('sell_value', 0)))
    fii_net  = _f(fii.get('netValue',  fii.get('net_value',  fii_buy - fii_sell)))
    dii_buy = dii_sell = dii_net = 0.0
    if dii:
        dii_buy  = _f(dii.get('buyValue',  0))
        dii_sell = _f(dii.get('sellValue', 0))
        dii_net  = _f(dii.get('netValue',  dii_buy - dii_sell))
    return [{'date': str(date.today()), 'fii_buy': fii_buy, 'fii_sell': fii_sell,
             'fii_net': fii_net, 'dii_buy': dii_buy, 'dii_sell': dii_sell,
             'dii_net': dii_net, 'combined_net': fii_net + dii_net}]

# ── Storage ───────────────────────────────────────────────
def store_records(records):
    if not records: return 0
    conn = get_connection()
    saved = 0
    for r in records:
        try:
            conn.execute("""
                INSERT OR REPLACE INTO fii_dii
                (date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net, combined_net)
                VALUES (?,?,?,?,?,?,?,?)
            """, (str(r.get('date', date.today())),
                  float(r.get('fii_buy', 0)), float(r.get('fii_sell', 0)),
                  float(r.get('fii_net', 0)), float(r.get('dii_buy', 0)),
                  float(r.get('dii_sell', 0)), float(r.get('dii_net', 0)),
                  float(r.get('combined_net', 0))))
            saved += 1
        except Exception:
            pass
    conn.commit(); conn.close()
    return saved

# ── Public API ────────────────────────────────────────────
def fetch_and_store(days=30):
    """Fetch latest FII/DII data and store in SQLite. Returns status dict."""
    create_fii_dii_table()

    live = _fetch_live()
    if live:
        parsed = _parse_live(live)
        saved  = store_records(parsed)
        if saved:
            return {'source': 'NSE Live API', 'rows_saved': saved, 'success': True}

    df = _fetch_csv(days)
    if not df.empty:
        records = []
        for _, row in df.iterrows():
            def col(*names):
                for n in names:
                    if n in row.index:
                        try: return float(str(row[n]).replace(',',''))
                        except: pass
                return 0.0
            fn = col('fii_net','fii/fpi_net_value')
            dn = col('dii_net','dii_net_value')
            records.append({
                'date': str(row.get('date', '')),
                'fii_buy': col('fii_buy','fii/fpi_buy_value'),
                'fii_sell': col('fii_sell','fii/fpi_sell_value'),
                'fii_net': fn, 'dii_buy': col('dii_buy','dii_buy_value'),
                'dii_sell': col('dii_sell','dii_sell_value'),
                'dii_net': dn, 'combined_net': fn + dn,
            })
        saved = store_records(records)
        if saved:
            return {'source': 'NSE CSV Archive', 'rows_saved': saved, 'success': True}

    return {'source': 'None', 'rows_saved': 0, 'success': False,
            'message': 'NSE may be rate-limiting. Cached data will be used if available.'}

def get_recent_data(days=15):
    """Returns the most recent N days from SQLite."""
    create_fii_dii_table()
    conn = get_connection()
    rows = conn.execute("""
        SELECT date, fii_buy, fii_sell, fii_net, dii_buy, dii_sell, dii_net, combined_net
        FROM fii_dii ORDER BY date DESC LIMIT ?
    """, (days,)).fetchall()
    conn.close()
    return [{'date': r[0], 'fii_buy': r[1], 'fii_sell': r[2], 'fii_net': r[3],
             'dii_buy': r[4], 'dii_sell': r[5], 'dii_net': r[6], 'combined_net': r[7]}
            for r in rows]

def get_flow_signal(days_avg=5):
    """Returns FII/DII signal + modifier for the ensemble."""
    create_fii_dii_table()
    recent = get_recent_data(days=days_avg)
    if not recent:
        return {'signal': 'NEUTRAL', 'modifier': 1.0, 'fii_net_avg': 0.0,
                'dii_net_avg': 0.0, 'combined_avg': 0.0, 'days_used': 0,
                'message': 'No FII/DII data. Run: python fetch_fii_dii.py'}

    fii_avg  = sum(r['fii_net']      for r in recent) / len(recent)
    dii_avg  = sum(r['dii_net']      for r in recent) / len(recent)
    comb_avg = sum(r['combined_net'] for r in recent) / len(recent)
    score    = fii_avg + dii_avg * 0.4

    if score > 2000:   signal, modifier = 'STRONGLY BULLISH', 1.08
    elif score > 500:  signal, modifier = 'BULLISH',          1.04
    elif score < -2000: signal, modifier = 'STRONGLY BEARISH', 0.92
    elif score < -500:  signal, modifier = 'BEARISH',          0.96
    else:              signal, modifier = 'NEUTRAL',           1.0

    return {
        'signal': signal, 'modifier': round(modifier, 4),
        'fii_net_avg': round(fii_avg, 2), 'dii_net_avg': round(dii_avg, 2),
        'combined_avg': round(comb_avg, 2), 'days_used': len(recent),
    }
