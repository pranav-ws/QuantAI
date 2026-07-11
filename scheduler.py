"""
scheduler.py — QuantAI Daily Automation Engine
Run this once and it handles EVERYTHING automatically:

  06:00 AM  →  Refresh stock data (pipeline.py)
  09:00 AM  →  Send morning briefing via Telegram
  03:45 PM  →  Run post-market paper trading scan
  04:00 PM  →  Send signal alerts via Telegram
  Friday 5PM→  Weekly performance summary

Start with:  python scheduler.py
Keep it running (minimise the terminal window, or use setup_autostart.bat
to make Windows launch it automatically on boot).

All output is written to data/scheduler.log
"""

import schedule# type: ignore
import time
import subprocess
import sys
import os
import json
import logging
from datetime import datetime, date

# ── Logging setup ─────────────────────────────────────────
os.makedirs("data", exist_ok=True)
LOG_FILE = os.path.join("data", "scheduler.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("QuantAI")

# ── Helpers ───────────────────────────────────────────────

def _python():
    """Returns the correct Python executable for this venv."""
    return sys.executable

def _run_script(script: str, label: str) -> bool:
    """Runs a Python script, logs output, returns True on success."""
    log.info(f"▶  Starting: {label}")
    start = time.time()
    try:
        result = subprocess.run(
            [_python(), script],
            capture_output=True, text=True, timeout=1800   # 30-min max
        )
        elapsed = time.time() - start
        if result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                log.info(f"   {line}")
        if result.stderr.strip():
            for line in result.stderr.strip().splitlines():
                if "WARNING" in line.upper() or "UserWarning" in line:
                    log.warning(f"   {line}")
                else:
                    log.error(f"   {line}")
        if result.returncode == 0:
            log.info(f"✅ Finished: {label}  ({elapsed:.1f}s)")
            _db_log(label, "SUCCESS", f"Completed in {elapsed:.1f}s")
            return True
        else:
            log.error(f"❌ Failed:   {label}  (exit {result.returncode})")
            _db_log(label, "ERROR", f"Exit {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"⏰ Timeout:  {label} (>30 min)")
        _db_log(label, "TIMEOUT", "Exceeded 30 minutes")
        return False
    except Exception as e:
        log.error(f"💥 Crash:    {label} — {e}")
        _db_log(label, "CRASH", str(e))
        return False

def _db_log(job: str, status: str, message: str = ""):
    """Writes a record to the scheduler_log table."""
    try:
        import sqlite3
        db = os.path.join("data", "quantai.db")
        if os.path.exists(db):
            conn = sqlite3.connect(db)
            conn.execute(
                "INSERT INTO scheduler_log (job,status,message) VALUES (?,?,?)",
                (job, status, message)
            )
            conn.commit()
            conn.close()
    except Exception:
        pass    # don't crash the scheduler over a log write

def _is_weekday() -> bool:
    return date.today().weekday() < 5   # Mon=0 … Fri=4

def _load_trade_log():
    p = os.path.join("data", "paper_trades.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            pass
    return []

# ── Scheduled Jobs ────────────────────────────────────────

def job_refresh_data():
    """06:00 AM — Pull latest NSE data (pipeline.py)."""
    if not _is_weekday():
        log.info("⏭  Skipped: refresh_data (weekend)")
        return
    log.info("=" * 58)
    log.info("📡  JOB: Daily Data Refresh")
    _run_script("pipeline.py", "pipeline.py")

def job_morning_briefing():
    """09:00 AM — Send Telegram morning briefing."""
    if not _is_weekday():
        return
    log.info("=" * 58)
    log.info("☀️  JOB: Morning Briefing")
    try:
        from src.data_collector import STOCK_UNIVERSE
        from src.alerts import send_morning_briefing
        import glob
        rf_count  = len(glob.glob("models/*_rf_model.pkl"))
        xgb_count = len(glob.glob("models/*_xgb_model.pkl"))
        seq_count = len(glob.glob("models/*_seqnn_model.pkl"))
        models_ready = max(rf_count, xgb_count, seq_count)
        send_morning_briefing(len(STOCK_UNIVERSE), models_ready)
        log.info("☀️  Morning briefing sent to Telegram")
        _db_log("morning_briefing", "SUCCESS")
    except Exception as e:
        log.error(f"Morning briefing failed: {e}")
        _db_log("morning_briefing", "ERROR", str(e))

def job_paper_trade():
    """03:45 PM — Run post-market paper trading scan."""
    if not _is_weekday():
        log.info("⏭  Skipped: paper_trade (weekend)")
        return
    log.info("=" * 58)
    log.info("📊  JOB: Post-Market Paper Trade Scan")
    _run_script("paper_trade.py", "paper_trade.py")

def job_signal_alert():
    """04:00 PM — Send evening summary to Telegram."""
    if not _is_weekday():
        return
    log.info("=" * 58)
    log.info("🔔  JOB: Evening Signal Alert")
    try:
        import time as _time
        start = _time.time()
        from src.paper_trader import get_signal
        from src.data_collector import STOCK_UNIVERSE
        from src.alerts import send_evening_summary

        signals, skipped = [], 0
        for ticker in STOCK_UNIVERSE:
            pred, conf, price, model = get_signal(ticker)
            if pred == 1:
                signals.append({"ticker": ticker, "price": price, "confidence": conf})
            else:
                skipped += 1

        elapsed = _time.time() - start
        send_evening_summary(signals, skipped, elapsed)
        log.info(f"🔔  Evening summary sent — {len(signals)} BUY(s), {skipped} SKIPs")
        _db_log("signal_alert", "SUCCESS", f"{len(signals)} signals")
    except Exception as e:
        log.error(f"Signal alert failed: {e}")
        _db_log("signal_alert", "ERROR", str(e))
        try:
            from src.alerts import send_error_alert
            send_error_alert("signal_alert", str(e))
        except Exception:
            pass

def job_weekly_summary():
    """Friday 05:00 PM — Send weekly performance summary."""
    if date.today().weekday() != 4:   # 4 = Friday
        return
    log.info("=" * 58)
    log.info("📊  JOB: Weekly Performance Summary")
    try:
        from src.alerts import send_weekly_summary
        trade_log = _load_trade_log()
        send_weekly_summary(trade_log)
        log.info("📊  Weekly summary sent to Telegram")
        _db_log("weekly_summary", "SUCCESS", f"{len(trade_log)} total trades")
    except Exception as e:
        log.error(f"Weekly summary failed: {e}")
        _db_log("weekly_summary", "ERROR", str(e))

def job_cleanup():
    """Midnight — clean up expired tokens and old log entries."""
    log.info("🧹  JOB: Nightly Cleanup")
    try:
        from src.auth import purge_expired_tokens
        purge_expired_tokens()
        log.info("🧹  Expired sessions purged")
        _db_log("cleanup", "SUCCESS")
    except Exception as e:
        log.error(f"Cleanup failed: {e}")



def job_tail_risk_check():
    """08:30 AM — Run tail risk scan and alert if elevated."""
    if not _is_weekday():
        return
    log.info("=" * 58)
    log.info("🔴  JOB: Tail Risk Monitor Scan")
    try:
        from src.tail_risk import TailRiskMonitor, LEVEL_NORMAL
        from src.alerts import send_tail_risk_alert

        monitor = TailRiskMonitor()
        report  = monitor.scan_from_db(period_days=252)

        log.info(f"🔴  TRI={report.tri:.3f}  Level={report.level}  "
                 f"Halt={report.halt_trading}")

        if report.level != LEVEL_NORMAL:
            send_tail_risk_alert(report)

        _db_log("tail_risk_check", "SUCCESS",
                f"TRI={report.tri:.3f} level={report.level}")
    except Exception as e:
        log.error(f"Tail risk check failed: {e}")
        _db_log("tail_risk_check", "ERROR", str(e))

# ── Schedule Registration ─────────────────────────────────

def setup_schedule():
    """Registers all jobs with the schedule library."""
    # IST times — your machine must be in IST (India Standard Time)
    schedule.every().day.at("06:00").do(job_refresh_data)
    schedule.every().day.at("09:00").do(job_morning_briefing)
    schedule.every().day.at("15:45").do(job_paper_trade)
    schedule.every().day.at("16:00").do(job_signal_alert)
    schedule.every().day.at("17:00").do(job_weekly_summary)
    schedule.every().day.at("00:05").do(job_cleanup)
    schedule.every().day.at("08:30").do(job_tail_risk_check)

    log.info("⏰  Scheduled jobs registered:")
    log.info("   06:00 AM  → Data refresh (pipeline)")
    log.info("   09:00 AM  → Morning briefing (Telegram)")
    log.info("   03:45 PM  → Post-market scan (paper_trade)")
    log.info("   04:00 PM  → Signal alert (Telegram)")
    log.info("   05:00 PM  → Weekly summary (Fridays only)")
    log.info("   12:05 AM  → Nightly cleanup")

# ── Main loop ─────────────────────────────────────────────

def main():
    log.info("=" * 58)
    log.info("  QuantAI Scheduler v1.0")
    log.info(f"  Started: {datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    log.info("=" * 58)

    # Ensure DB tables exist
    try:
        from src.database import create_tables
        create_tables()
    except Exception as e:
        log.warning(f"DB init warning: {e}")

    setup_schedule()

    # Send startup notification
    try:
        from src.alerts import send_scheduler_started
        send_scheduler_started()
    except Exception:
        pass

    log.info("")
    log.info("🟢  Scheduler is RUNNING. Do not close this window.")
    log.info(f"   Next job: {schedule.next_run()}")
    log.info("   Press Ctrl+C to stop.")
    log.info("")

    while True:
        try:
            schedule.run_pending()
            time.sleep(30)     # check every 30 seconds
        except KeyboardInterrupt:
            log.info("🛑  Scheduler stopped by user.")
            break
        except Exception as e:
            log.error(f"Main loop error: {e}")
            time.sleep(60)     # wait 1 min then try again

if __name__ == "__main__":
    main()
