"""
src/auth.py — QuantAI multi-user authentication
Zero external dependencies: uses only Python stdlib (hashlib, secrets, sqlite3).
Token-based sessions stored in SQLite.  Works on Python 3.14+.
"""
import hashlib, secrets, sqlite3, os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'quantai.db')
TOKEN_TTL_DAYS = 7  # sessions expire after 7 days

# ── Password helpers ──────────────────────────────────────

def hash_password(password: str, salt: str = None):
    """Returns (hash, salt). Pass salt to verify an existing hash."""
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()
    return digest, salt

def verify_password(plain: str, stored_hash: str, salt: str) -> bool:
    digest, _ = hash_password(plain, salt)
    return digest == stored_hash

# ── Session helpers ───────────────────────────────────────

def create_token(user_id: int) -> str:
    """Generates a secure session token and saves it to the DB."""
    token    = secrets.token_hex(32)
    expires  = (datetime.now() + timedelta(days=TOKEN_TTL_DAYS)).isoformat()
    conn     = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (token, user_id, expires_at) VALUES (?,?,?)",
        (token, user_id, expires)
    )
    conn.commit()
    conn.close()
    return token

def validate_token(token: str):
    """Returns user_id if token is valid and not expired, else None.

    Never raises — a missing DB file, missing 'data/' folder, or missing
    'sessions' table (e.g. before pipeline.py has been run, or on a fresh
    checkout) must not crash every request that carries a stale
    Authorization header. Those cases are treated the same as "no valid
    session": return None.
    """
    if not token:
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        row  = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token=?", (token,)
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    try:
        user_id, expires_at = row
        if datetime.fromisoformat(expires_at) < datetime.now():
            return None          # expired
        return user_id
    except Exception:
        return None

def revoke_token(token: str):
    """Deletes a session token (logout)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()

def purge_expired_tokens():
    """Deletes expired tokens (call from scheduler nightly)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE expires_at < ?",
                 (datetime.now().isoformat(),))
    conn.commit()
    conn.close()

# ── User CRUD ─────────────────────────────────────────────

def register_user(username: str, email: str, password: str, is_admin: bool = False):
    """
    Creates a new user.
    Returns (True, user_id) on success, (False, error_message) on failure.
    """
    pw_hash, salt = hash_password(password)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.execute(
            "INSERT INTO users (username,email,password_hash,salt,is_admin) VALUES (?,?,?,?,?)",
            (username.strip(), email.strip().lower(), pw_hash, salt, int(is_admin))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return True, user_id
    except sqlite3.IntegrityError as e:
        conn.close()
        if "username" in str(e):
            return False, "Username already taken"
        if "email" in str(e):
            return False, "Email already registered"
        return False, str(e)

def login_user(username_or_email: str, password: str):
    """
    Validates credentials and returns (True, token, user_dict) or (False, error, None).
    """
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT id,username,email,password_hash,salt,is_admin FROM users "
        "WHERE username=? OR email=?",
        (username_or_email, username_or_email.lower())
    ).fetchone()
    conn.close()

    if not row:
        return False, "User not found", None

    uid, username, email, pw_hash, salt, is_admin = row
    if not verify_password(password, pw_hash, salt):
        return False, "Incorrect password", None

    token = create_token(uid)
    return True, token, {
        "id": uid, "username": username, "email": email, "is_admin": bool(is_admin)
    }

def get_user_by_id(user_id: int):
    """Returns user dict or None."""
    conn = sqlite3.connect(DB_PATH)
    row  = conn.execute(
        "SELECT id,username,email,is_admin,created_at FROM users WHERE id=?",
        (user_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"id": row[0], "username": row[1], "email": row[2],
            "is_admin": bool(row[3]), "created_at": row[4]}

def get_all_users():
    """Returns all users (admin use)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id,username,email,is_admin,created_at FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return [{"id":r[0],"username":r[1],"email":r[2],"is_admin":bool(r[3]),"created_at":r[4]}
            for r in rows]

def count_users():
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return n

# ── Watchlist helpers ─────────────────────────────────────

def get_watchlist(user_id: int):
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT ticker, added_at FROM watchlists WHERE user_id=? ORDER BY added_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return [{"ticker": r[0], "added_at": r[1]} for r in rows]

def add_to_watchlist(user_id: int, ticker: str):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlists (user_id,ticker) VALUES (?,?)",
            (user_id, ticker.upper())
        )
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False

def remove_from_watchlist(user_id: int, ticker: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "DELETE FROM watchlists WHERE user_id=? AND ticker=?",
        (user_id, ticker.upper())
    )
    conn.commit()
    conn.close()

# ── Per-user Paper Trades ──────────────────────────────────

def add_user_trade(user_id: int, trade: dict):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO user_trades "
        "(user_id,ticker,shares,buy_price,stop_loss,trade_value,signal_date,confidence) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (user_id, trade['ticker'], trade['shares'], trade['price'],
         trade.get('stop_loss',0), trade.get('trade_value',0),
         trade.get('date',''), trade.get('confidence',0))
    )
    conn.commit()
    conn.close()

def get_user_trades(user_id: int):
    conn  = sqlite3.connect(DB_PATH)
    rows  = conn.execute(
        "SELECT id,ticker,shares,buy_price,stop_loss,trade_value,"
        "signal_date,confidence,status,sell_price,pnl,closed_at "
        "FROM user_trades WHERE user_id=? ORDER BY id DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    keys  = ["id","ticker","shares","buy_price","stop_loss","trade_value",
             "signal_date","confidence","status","sell_price","pnl","closed_at"]
    return [dict(zip(keys, r)) for r in rows]
