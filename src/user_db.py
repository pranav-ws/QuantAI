"""
src/user_db.py
User management database operations.
Tables: users, user_portfolios, user_trades
"""
import sqlite3
from datetime import date
from src.database import get_connection

# ── Table creation ────────────────────────────────────────
def create_user_tables():
    conn   = get_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    DEFAULT "user",
            created_at    TEXT    DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_portfolios (
            user_id    INTEGER PRIMARY KEY,
            capital    REAL    DEFAULT 100000,
            peak       REAL    DEFAULT 100000,
            start_date TEXT    DEFAULT CURRENT_DATE,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            ticker      TEXT    NOT NULL,
            price       REAL,
            shares      INTEGER,
            confidence  REAL,
            stop_loss   REAL,
            trade_value REAL,
            model_type  TEXT,
            sentiment   REAL    DEFAULT 0,
            date        TEXT    DEFAULT CURRENT_DATE,
            status      TEXT    DEFAULT "OPEN",
            pnl         REAL    DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

# ── User CRUD ─────────────────────────────────────────────
def create_user(username: str, email: str, password_hash: str):
    """Creates a new user. First user gets admin role."""
    conn   = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM users")
        role = 'admin' if cursor.fetchone()[0] == 0 else 'user'

        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?,?,?,?)",
            (username.lower().strip(), email.lower().strip(), password_hash, role)
        )
        user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO user_portfolios (user_id, start_date) VALUES (?,?)",
            (user_id, str(date.today()))
        )
        conn.commit()
        return {"success": True, "user_id": user_id, "role": role}
    except sqlite3.IntegrityError as e:
        err = str(e)
        if 'username' in err:
            return {"success": False, "error": "Username already taken"}
        if 'email' in err:
            return {"success": False, "error": "Email already registered"}
        return {"success": False, "error": err}
    finally:
        conn.close()

def get_user_by_username(username: str):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, password_hash, role FROM users WHERE username=?",
        (username.lower().strip(),)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2],
                "password_hash": row[3], "role": row[4]}
    return None

def get_user_by_id(user_id: int):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, email, role, created_at FROM users WHERE id=?",
        (user_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "email": row[2],
                "role": row[3], "created_at": row[4]}
    return None

def get_all_users():
    """Admin only — returns all users."""
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT u.id, u.username, u.email, u.role, u.created_at, "
        "  COALESCE(p.capital, 100000) as capital, "
        "  (SELECT COUNT(*) FROM user_trades t WHERE t.user_id=u.id) as trades "
        "FROM users u LEFT JOIN user_portfolios p ON p.user_id=u.id "
        "ORDER BY u.created_at"
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "email": r[2], "role": r[3],
             "created_at": r[4], "capital": r[5], "trades": r[6]}
            for r in rows]

# ── Portfolio ─────────────────────────────────────────────
def get_user_portfolio(user_id: int):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT capital, peak, start_date FROM user_portfolios WHERE user_id=?",
        (user_id,)
    )
    row = cursor.fetchone()
    if not row:
        cursor.execute(
            "INSERT INTO user_portfolios (user_id, start_date) VALUES (?,?)",
            (user_id, str(date.today()))
        )
        conn.commit()
        row = (100000.0, 100000.0, str(date.today()))
    conn.close()
    return {"capital": row[0], "peak": row[1], "start_date": row[2]}

def update_user_capital(user_id: int, capital: float):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE user_portfolios SET capital=?, peak=MAX(peak,?) WHERE user_id=?",
        (capital, capital, user_id)
    )
    conn.commit()
    conn.close()

# ── Trades ────────────────────────────────────────────────
def save_user_trade(user_id: int, trade: dict):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_trades "
        "(user_id,ticker,price,shares,confidence,stop_loss,trade_value,model_type,sentiment,date,status) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (user_id, trade['ticker'], trade['price'], trade.get('shares', 0),
         trade.get('confidence', 0), trade.get('stop_loss', 0),
         trade.get('trade_value', 0), trade.get('model_type', 'Ensemble'),
         trade.get('sentiment', 0), trade.get('date', str(date.today())), 'OPEN')
    )
    conn.commit()
    conn.close()

def get_user_trades(user_id: int, limit: int = 20):
    conn   = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker,price,shares,confidence,stop_loss,trade_value,model_type,date,status,pnl "
        "FROM user_trades WHERE user_id=? ORDER BY date DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"ticker": r[0], "price": r[1], "shares": r[2], "confidence": r[3],
             "stop_loss": r[4], "trade_value": r[5], "model_type": r[6],
             "date": r[7], "status": r[8], "pnl": r[9]}
            for r in rows]
