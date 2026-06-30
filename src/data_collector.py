import yfinance as yf# type: ignore
import pandas as pd# type: ignore
from datetime import datetime, timedelta
from src.database import insert_stock, insert_prices, get_latest_date

# Full Nifty 50 stock universe — 50 stocks across 16 sectors
STOCK_UNIVERSE = {

    # ── Banking (6) ──────────────────────────────────────
    'HDFCBANK.NS'  : ('HDFC Bank',                 'Banking'),
    'ICICIBANK.NS' : ('ICICI Bank',                'Banking'),
    'SBIN.NS'      : ('State Bank of India',       'Banking'),
    'KOTAKBANK.NS' : ('Kotak Mahindra Bank',       'Banking'),
    'AXISBANK.NS'  : ('Axis Bank',                 'Banking'),
    'INDUSINDBK.NS': ('IndusInd Bank',             'Banking'),

    # ── Technology (5) ───────────────────────────────────
    'TCS.NS'       : ('Tata Consultancy Services', 'Technology'),
    'INFY.NS'      : ('Infosys',                   'Technology'),
    'HCLTECH.NS'   : ('HCL Technologies',          'Technology'),
    'WIPRO.NS'     : ('Wipro',                     'Technology'),
    'TECHM.NS'     : ('Tech Mahindra',             'Technology'),

    # ── Energy (3) ───────────────────────────────────────
    'RELIANCE.NS'  : ('Reliance Industries',       'Energy'),
    'ONGC.NS'      : ('Oil & Natural Gas Corp',    'Energy'),
    'BPCL.NS'      : ('Bharat Petroleum',          'Energy'),

    # ── Power (3) ────────────────────────────────────────
    'NTPC.NS'      : ('NTPC',                      'Power'),
    'POWERGRID.NS' : ('Power Grid Corporation',    'Power'),
    'COALINDIA.NS' : ('Coal India',                'Power'),

    # ── FMCG (5) ─────────────────────────────────────────
    'HINDUNILVR.NS': ('Hindustan Unilever',        'FMCG'),
    'ITC.NS'       : ('ITC Limited',               'FMCG'),
    'BRITANNIA.NS' : ('Britannia Industries',      'FMCG'),
    'NESTLEIND.NS' : ('Nestle India',              'FMCG'),
    'TATACONSUM.NS': ('Tata Consumer Products',    'FMCG'),

    # ── Automobile (6) ───────────────────────────────────
    'BAJAJ-AUTO.NS': ('Bajaj Auto',                'Automobile'),
    'HEROMOTOCO.NS': ('Hero MotoCorp',             'Automobile'),
    'MARUTI.NS'    : ('Maruti Suzuki',             'Automobile'),
    'TATAMOTORS.BO': ('Tata Motors',               'Automobile'),
    'EICHERMOT.NS' : ('Eicher Motors',             'Automobile'),
    'M&M.NS'       : ('Mahindra & Mahindra',       'Automobile'),

    # ── Pharma (4) ───────────────────────────────────────
    'SUNPHARMA.NS' : ('Sun Pharmaceutical',        'Pharma'),
    'DRREDDY.NS'   : ("Dr Reddy's Laboratories",  'Pharma'),
    'CIPLA.NS'     : ('Cipla',                     'Pharma'),
    'DIVISLAB.NS'  : ("Divi's Laboratories",       'Pharma'),

    # ── Metals (3) ───────────────────────────────────────
    'TATASTEEL.NS' : ('Tata Steel',                'Metals'),
    'HINDALCO.NS'  : ('Hindalco Industries',       'Metals'),
    'JSWSTEEL.NS'  : ('JSW Steel',                 'Metals'),

    # ── Cement (2) ───────────────────────────────────────
    'ULTRACEMCO.NS': ('UltraTech Cement',          'Cement'),
    'GRASIM.NS'    : ('Grasim Industries',         'Cement'),

    # ── Consumer & Retail (3) ────────────────────────────
    'ASIANPAINT.NS': ('Asian Paints',              'Consumer'),
    'TITAN.NS'     : ('Titan Company',             'Consumer'),
    'TRENT.NS'     : ('Trent',                     'Consumer'),

    # ── Healthcare (1) ───────────────────────────────────
    'APOLLOHOSP.NS': ('Apollo Hospitals',          'Healthcare'),

    # ── Insurance (2) ────────────────────────────────────
    'HDFCLIFE.NS'  : ('HDFC Life Insurance',       'Insurance'),
    'SBILIFE.NS'   : ('SBI Life Insurance',        'Insurance'),

    # ── Financial Services (3) ───────────────────────────
    'BAJFINANCE.NS': ('Bajaj Finance',             'Financial Services'),
    'BAJAJFINSV.NS': ('Bajaj Finserv',             'Financial Services'),
    'SHRIRAMFIN.NS': ('Shriram Finance',           'Financial Services'),

    # ── Infrastructure (2) ───────────────────────────────
    'LT.NS'        : ('Larsen & Toubro',           'Infrastructure'),
    'ADANIPORTS.NS': ('Adani Ports',               'Infrastructure'),

    # ── Telecom (1) ──────────────────────────────────────
    'BHARTIARTL.NS': ('Bharti Airtel',             'Telecom'),

    # ── Conglomerate (1) ─────────────────────────────────
    'ADANIENT.NS'  : ('Adani Enterprises',         'Conglomerate'),
}

def fetch_stock(ticker, name, sector, default_start='2022-01-01'):
    """Downloads data for one stock and saves it to the database."""

    # Smart update: only download data we don't already have
    latest = get_latest_date(ticker)
    if latest:
        start = str((datetime.strptime(latest, '%Y-%m-%d') + timedelta(days=1)).date())
        print(f"  🔄  {ticker:<20} Updating from {start}")
    else:
        start = default_start
        print(f"  📥  {ticker:<20} Fresh download from {start}")

    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)

        if df.empty:
            print(f"  ⚠️  {ticker:<20} No new data available")
            return 0

        # Flatten MultiIndex columns (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Save to DB
        insert_stock(ticker, name, sector)
        rows = insert_prices(ticker, df)
        print(f"  ✅  {ticker:<20} {rows} rows saved")
        return rows

    except Exception as e:
        print(f"  ❌  {ticker:<20} Error: {e}")
        return 0

def fetch_all_stocks():
    """Runs the full pipeline for every stock in our universe."""
    print("\n" + "=" * 52)
    print("   QuantAI — Nifty 50 Data Collection Pipeline")
    print("=" * 52)

    total = 0
    for ticker, (name, sector) in STOCK_UNIVERSE.items():
        total += fetch_stock(ticker, name, sector)

    print("=" * 52)
    print(f"   Pipeline complete — {total:,} total rows saved")
    print("=" * 52 + "\n")
