import sqlite3
import os

# Path to our database file
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'quantai.db')

def get_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)

def create_tables():
    """Creates all tables if they don't already exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Table 1: Stock metadata (name, sector)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            ticker    TEXT PRIMARY KEY,
            name      TEXT,
            sector    TEXT,
            added_on  TEXT DEFAULT (date('now'))
        )
    ''')

    # Table 2: Daily OHLCV price data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker    TEXT NOT NULL,
            date      TEXT NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    INTEGER,
            UNIQUE(ticker, date),
            FOREIGN KEY (ticker) REFERENCES stocks(ticker)
        )
    ''')

    # ── Multi-user tables ─────────────────────────────────

    # Table 3: User accounts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            is_admin      INTEGER DEFAULT 0,
            created_at    TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Idempotent column add for Telegram bot linking — ALTER TABLE ADD
    # COLUMN fails if the column already exists, so this is wrapped so
    # re-running create_tables() (happens on every server startup) never
    # crashes on the second+ run.
    try:
        cursor.execute("ALTER TABLE users ADD COLUMN telegram_chat_id TEXT")
    except sqlite3.OperationalError:
        pass   # column already exists

    # Short-lived tokens for the Telegram "Add Telegram Bot" connect flow —
    # see /telegram/connect-link and /telegram/webhook in src/api.py.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telegram_link_tokens (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Table 4: Login sessions (token-based auth, no external libs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Table 5: Per-user watchlists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS watchlists (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id   INTEGER NOT NULL,
            ticker    TEXT NOT NULL,
            added_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, ticker),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Table 6: Per-user paper trades
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            ticker      TEXT NOT NULL,
            shares      INTEGER NOT NULL,
            buy_price   REAL NOT NULL,
            stop_loss   REAL,
            trade_value REAL,
            signal_date TEXT,
            confidence  REAL,
            status      TEXT DEFAULT 'OPEN',
            sell_price  REAL,
            pnl         REAL,
            closed_at   TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # Table 7: Scheduler log — records of each automated run
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scheduler_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            job        TEXT NOT NULL,
            status     TEXT NOT NULL,
            message    TEXT,
            ran_at     TEXT DEFAULT (datetime('now'))
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database ready (users + scheduler log added)!")

def insert_stock(ticker, name, sector):
    """Adds a stock entry (skips if already exists)."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR IGNORE INTO stocks (ticker, name, sector)
        VALUES (?, ?, ?)
    ''', (ticker, name, sector))
    conn.commit()
    conn.close()

def insert_prices(ticker, df):
    """Saves price rows from a DataFrame into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    rows_saved = 0

    for date, row in df.iterrows():
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO prices
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                str(date.date()),
                round(float(row['Open']),   2),
                round(float(row['High']),   2),
                round(float(row['Low']),    2),
                round(float(row['Close']),  2),
                int(row['Volume'])
            ))
            rows_saved += 1
        except Exception as e:
            print(f"  ⚠️  Skipped {date}: {e}")

    conn.commit()
    conn.close()
    return rows_saved

def get_latest_date(ticker):
    """Returns the most recent date we have stored for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT MAX(date) FROM prices WHERE ticker = ?', (ticker,))
    result = cursor.fetchone()[0]
    conn.close()
    return result

def get_prices(ticker, limit=10):
    """Returns the most recent price rows for a ticker."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, open, high, low, close, volume
        FROM prices WHERE ticker = ?
        ORDER BY date DESC LIMIT ?
    ''', (ticker, limit))
    rows = cursor.fetchall()
    conn.close()
    return rows

def log_scheduler_run(job: str, status: str, message: str = ""):
    """Records an automated scheduler run in the database."""
    conn = get_connection()
    conn.execute(
        "INSERT INTO scheduler_log (job, status, message) VALUES (?,?,?)",
        (job, status, message)
    )
    conn.commit()
    conn.close()
