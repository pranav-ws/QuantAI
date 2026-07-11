"""
src/alerts.py — QuantAI Telegram alerts (enhanced)
Supports: signal alerts, morning briefing, evening summary, weekly report.
Sends to every user who has connected their own Telegram account (see
/telegram/connect-link in src/api.py), not just a single hardcoded chat.
"""

from datetime import datetime, date
import os
import requests# type: ignore

# ── Bot token from environment, never hardcoded ────────────
# This used to have a real token and chat_id hardcoded directly in this
# file, which is a serious problem the moment this repo is pushed to
# GitHub — anyone who sees it can send messages as your bot or read your
# alerts. If you're migrating from an older version of this file that had
# a token here, revoke it via @BotFather ("Revoke current token") and
# generate a fresh one, then put ONLY the new one in your .env file as
# TELEGRAM_BOT_TOKEN — never commit it to source control.
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# ── Internal check ────────────────────────────────────────

def _configured() -> bool:
    return bool(BOT_TOKEN)


def _send_to(chat_id: str, message: str) -> bool:
    """Sends a message to one specific chat_id."""
    if not BOT_TOKEN or not chat_id:
        return False
    try:
        url      = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        response = requests.post(url, json={
            "chat_id"                  : chat_id,
            "text"                     : message,
            "parse_mode"               : "Markdown",
            "disable_web_page_preview" : True
        }, timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"  ⚠️  Telegram send to {chat_id} failed: {e}")
        return False


def _broadcast(message: str):
    """Sends a message to every user who has connected their Telegram account."""
    if not _configured():
        return
    try:
        from src.user_db import get_all_telegram_chat_ids
        chat_ids = get_all_telegram_chat_ids()
    except Exception as e:
        print(f"  ⚠️  Could not load connected Telegram users: {e}")
        return
    for chat_id in chat_ids:
        _send_to(chat_id, message)


def _run(message: str):
    """Back-compat name used throughout this file — now broadcasts to all
    connected users instead of a single hardcoded chat_id."""
    _broadcast(message)

def send_message(text: str) -> bool:
    """Public wrapper — broadcasts to all connected users. Used by
    pairs_trade.py and other modules that just want to send one message."""
    _broadcast(text)
    return True

# ── Alert types ───────────────────────────────────────────

def send_signal_alert(signals: list):
    """Sends a Telegram message listing today's BUY signals."""
    if not signals:
        return
    msg  = f"🤖 *QuantAI Signal Alert* — {date.today().strftime('%d %b %Y')}\n\n"
    for s in signals:
        tkr  = s['ticker'].replace('.NS', '')
        conf = s['confidence'] * 100
        msg += f"🟢 *{tkr}* — BUY\n"
        msg += f"   Price: ₹{s['price']:,.2f}  |  Confidence: {conf:.1f}%\n"

        # Show detected patterns
        patterns = s.get('patterns', [])
        for pat in patterns[:2]:
            msg += f"   📐 {pat}\n"
        msg += "\n"

    msg += "_QuantAI Daily Automation_"
    _run(msg)


def send_factor_alert(rankings: list):
    """Sends weekly factor model update."""
    if not rankings:
        return
    today   = date.today().strftime('%d %b %Y')
    top5    = [r for r in rankings if r['rank'] <= 5]
    bottom5 = sorted(rankings,
                     key=lambda x: x['rank'],
                     reverse=True)[:5]
    msg  = f"📊 *QuantAI Factor Model* — {today}\n\n"
    msg += f"🏆 *Top 5 Factor Stocks:*\n"
    for r in top5:
        msg += (f"   #{r['rank']} *{r['ticker'].replace('.NS','')}*"
                f" — Score: {r['composite_score']:.3f}\n"
                f"   V:{r['value_score'] or 'N/A':.3f}  "
                f"M:{r['momentum_score'] or 'N/A':.3f}  "
                f"Q:{r['quality_score'] or 'N/A':.3f}\n")
    msg += f"\n🔴 *Avoid (Bottom 5):*\n"
    for r in bottom5:
        msg += (f"   #{r['rank']} "
                f"{r['ticker'].replace('.NS','')} "
                f"({r['composite_score']:.3f})\n")
    msg += "\n_Run weekly — fundamentals don't change daily_"
    _run(msg)


def send_morning_briefing(total_stocks: int, models_ready: int):
    """
    Morning pre-market briefing — sent ~9 AM IST (before NSE open at 9:15 AM).
    Lets you know the system is live and how many models are loaded.
    """
    now  = datetime.now().strftime("%I:%M %p")
    msg  = (
        f"☀️ *QuantAI Morning Briefing* — {date.today().strftime('%d %b %Y')}\n\n"
        f"🕘 System running at *{now} IST*\n"
        f"📊 Monitoring *{total_stocks}* Nifty 50 stocks\n"
        f"🤖 *{models_ready}* ML models loaded (RF + XGBoost + SeqNN Ensemble)\n\n"
        f"📅 NSE opens at 09:15 AM — signals will be generated after close\n"
        f"_Post-market scan runs at 3:45 PM IST automatically_"
    )
    _run(msg)


def send_evening_summary(signals: list, skipped: int, scan_time_sec: float = 0):
    """
    Evening post-market summary — sent ~4 PM IST after paper_trade.py runs.
    Shows full signal table and summary stats.
    """
    n_buy  = len(signals)
    n_skip = skipped
    total  = n_buy + n_skip
    today  = date.today().strftime('%d %b %Y')

    if n_buy == 0:
        msg = (
            f"🌙 *QuantAI Evening Summary* — {today}\n\n"
            f"💤 *No BUY signals* today out of {total} stocks scanned\n"
            f"📊 All {n_skip} stocks SKIP — staying in cash is the right move\n"
            f"_Ensemble confidence threshold: 58%_"
        )
    else:
        msg  = f"🌙 *QuantAI Evening Summary* — {today}\n\n"
        msg += f"✅ *{n_buy} BUY signal(s)* from {total} stocks scanned\n\n"
        for s in signals[:5]:   # cap at 5 to keep message short
            tkr  = s['ticker'].replace('.NS', '')
            conf = s.get('confidence', 0) * 100
            msg += f"🟢 *{tkr}* — ₹{s['price']:,.0f}  |  {conf:.1f}%\n"

            # Show detected patterns under each signal
            patterns = s.get('patterns', [])
            for pat in patterns[:2]:   # max 2 patterns per stock
                msg += f"   📐 {pat}\n"
        if n_buy > 5:
            msg += f"_...and {n_buy-5} more_\n"
        if scan_time_sec:
            msg += f"\n⏱ Scan completed in {scan_time_sec:.1f}s"

    _run(msg)


def send_weekly_summary(trade_log: list):
    """
    Weekly performance summary — sent every Friday ~5 PM IST.
    Includes win/loss record, best/worst trades of the week.
    """
    today      = date.today().strftime('%d %b %Y')
    closed     = [t for t in trade_log if t.get('status') == 'CLOSED']
    open_t     = [t for t in trade_log if t.get('status') == 'OPEN']

    wins       = sum(1 for t in closed if (t.get('pnl') or 0) > 0)
    losses     = len(closed) - wins
    total_pnl  = sum((t.get('pnl') or 0) for t in closed)
    win_rate   = (wins / len(closed) * 100) if closed else 0

    msg = (
        f"📊 *QuantAI Weekly Summary* — {today}\n\n"
        f"📋 Open positions  : *{len(open_t)}*\n"
        f"✅ Closed trades   : *{len(closed)}*\n"
        f"🏆 Wins / Losses   : *{wins} / {losses}*\n"
        f"📈 Win rate        : *{win_rate:.1f}%*\n"
        f"💰 Total P&L       : *₹{total_pnl:+,.0f}*\n\n"
        f"_QuantAI Automated Weekly Report_"
    )
    _run(msg)


def send_var_alert(report: dict):
    """
    Sends daily VaR summary to Telegram.
    Called from paper_trade.py after each scan.
    """
    port_var = report.get('portfolio_var')
    capital  = report.get('capital', 100000)
    stress   = report.get('stress_tests', {})

    if port_var:
        var_inr = port_var['conservative_var_inr']
        var_pct = port_var['conservative_var_pct']
        n_pos   = port_var['n_positions']
        div_ben = port_var['diversification_benefit']

        if var_pct < 2:
            risk_emoji = '🟢'
            risk_label = 'LOW'
        elif var_pct < 4:
            risk_emoji = '🟡'
            risk_label = 'MEDIUM'
        else:
            risk_emoji = '🔴'
            risk_label = 'HIGH'

        msg = (
            f"📉 *QuantAI Risk Report*\n\n"
            f"{risk_emoji} Risk Level: *{risk_label}*\n\n"
            f"💼 Open Positions: *{n_pos}*\n"
            f"📊 Portfolio VaR (95%): *₹{var_inr:,.0f}* "
            f"({var_pct:.2f}%)\n"
            f"🛡 Diversification saved: *₹{div_ben:,.0f}*\n\n"
        )

        # Worst stress scenario
        if stress:
            worst = max(stress.items(),
                        key=lambda x: x[1]['loss_inr'])
            msg += (
                f"⚠️ Worst stress scenario:\n"
                f"   {worst[0]}: "
                f"*₹{worst[1]['loss_inr']:,.0f} loss*\n\n"
            )
        msg += "_QuantAI Risk Management_"

    else:
        msg = (
            f"📉 *QuantAI Risk Report*\n\n"
            f"💰 Capital: ₹{capital:,.0f}\n"
            f"📊 No open positions — "
            f"full capital in cash\n"
            f"✅ VaR: ₹0 (no market exposure)\n\n"
            f"_QuantAI Risk Management_"
        )

    _run(msg)

def send_regime_alert(regime_data: dict):
    """Sends market regime update to Telegram."""
    from src.regime_detector import REGIMES
    regime  = regime_data.get('regime', 'SIDEWAYS')
    comp    = regime_data.get('composite', 0.5)
    scores  = regime_data.get('scores', {})
    trading = regime_data.get('trading', {})
    emoji   = REGIMES[regime]['emoji']

    trend_pct = regime_data.get('details', {}).get(
        'trend', {}
    ).get('pct_above_ma200', 0)

    adv = regime_data.get('details', {}).get(
        'breadth', {}
    ).get('advancing', 0)

    msg = (
        f"📡 *QuantAI Market Regime*\n\n"
        f"{emoji} Current Regime: *{regime}*\n"
        f"📊 Composite Score: *{comp:.3f}*\n\n"
        f"5-Factor Scores:\n"
        f"  Trend      : {scores.get('trend', 0):.2f}\n"
        f"  Breadth    : {scores.get('breadth', 0):.2f} "
        f"({adv} stocks advancing)\n"
        f"  Momentum   : {scores.get('momentum', 0):.2f}\n"
        f"  Volatility : {scores.get('volatility', 0):.2f}\n"
        f"  RSI        : {scores.get('rsi', 0):.2f}\n\n"
        f"⚙️ Trading Parameters:\n"
        f"  Threshold  : *{trading.get('threshold', 0.58):.0%}*\n"
        f"  Max Trades : *{trading.get('max_positions', 4)}*\n\n"
        f"_{trading.get('description', '')}_"
    )
    _run(msg)

def send_rotation_alert(ranked_sectors: list):
    """Sends sector rotation update to Telegram."""
    if not ranked_sectors:
        return
    today = date.today().strftime('%d %b %Y')
    msg   = f"🔄 *QuantAI Sector Rotation*\n📅 {today}\n\n"
    msg  += f"🟢 *BUY Sectors (Top 3):*\n"
    for r in ranked_sectors:
        if r['tier'] == 'TOP':
            msg += (f"   #{r['rank']} *{r['sector']}* — "
                    f"{r['return_1m']:+.1f}% "
                    f"(Score: {r['composite_score']:.3f})\n")
    msg  += f"\n🔴 *AVOID Sectors (Bottom 3):*\n"
    for r in ranked_sectors:
        if r['tier'] == 'BOTTOM':
            msg += (f"   #{r['rank']} *{r['sector']}* — "
                    f"{r['return_1m']:+.1f}%\n")
    msg  += "\n_Rotate capital into top sectors_"
    _run(msg)


def send_scheduler_started():
    """Notification when the scheduler boots up (on machine restart etc.)."""
    msg = (
        f"🚀 *QuantAI Scheduler Started*\n\n"
        f"📅 {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n"
        f"⏰ Scheduled jobs:\n"
        f"   • 09:00 AM — Morning briefing\n"
        f"   • 06:00 AM — Data refresh (pipeline)\n"
        f"   • 03:45 PM — Post-market scan (paper trade)\n"
        f"   • 04:00 PM — Signal alert\n"
        f"   • Friday 05:00 PM — Weekly summary\n\n"
        f"_System is running — no manual action needed_"
    )
    _run(msg)


def send_error_alert(job: str, error: str):
    """Called when a scheduled job fails."""
    msg = (
        f"⚠️ *QuantAI Scheduler Error*\n\n"
        f"Job: `{job}`\n"
        f"Error: `{error[:200]}`\n"
        f"Time: {datetime.now().strftime('%d %b %Y, %I:%M %p')}\n\n"
        f"_Check data\\log.txt for details_"
    )
    _run(msg)


def send_tail_risk_alert(report):
    """
    Sends a Telegram alert when TRI is ELEVATED or above.
    Includes all detector signals and the recommendation.
    """
    from src.tail_risk import LEVEL_EMOJI
    emoji = LEVEL_EMOJI.get(report.level, '⚠️')
    today = date.today().strftime('%d %b %Y')

    msg = (
        f"{emoji} *QuantAI Tail Risk Alert* — {today}\n\n"
        f"*Level: {report.level}*  |  TRI = `{report.tri:.3f}`\n"
        f"Stocks scanned: {report.n_stocks}\n\n"
        f"*Component Scores:*\n"
        f"  Volatility    `{report.vol_score:.3f}`\n"
        f"  Fat-tail      `{report.kurtosis_score:.3f}`\n"
        f"  Neg. skew     `{report.skew_score:.3f}`\n"
        f"  Correlation   `{report.correlation_score:.3f}`\n"
        f"  Liquidity     `{report.liquidity_score:.3f}`\n"
        f"  VaR breaches  `{report.var_breach_count}/{report.n_stocks}`\n\n"
        f"*Recommendation:*\n_{report.recommendation}_\n\n"
        f"_QuantAI Black Swan Detector_"
    )
    _run(msg)


def send_risk_parity_alert(rp_result, signals: list):
    """
    Sends risk parity allocation summary to Telegram.
    Called after paper_trade.py completes its session.
    """
    today = date.today().strftime('%d %b %Y')
    lines = [
        f"⚖️ *QuantAI Risk Parity Allocation* — {today}\n",
        f"Positions: *{rp_result.n_stocks}*  |  "
        f"Deployed: ₹{rp_result.total_deployed:,.0f}  |  "
        f"Cash: ₹{rp_result.cash_remaining:,.0f}\n",
        f"Target daily risk/pos: ₹{rp_result.target_risk_per_pos:,.0f}\n\n",
    ]
    for s in signals:
        t   = s['ticker'].replace('.NS', '')
        vol = s.get('rp_daily_vol', 0)
        w   = s.get('rp_weight', 0) * 100
        lines.append(
            f"  {t}: {s['shares']} shares  {w:.1f}% capital  σ={vol:.2f}%/day\n"
        )
    lines.append("\n_Risk Parity: equal risk, not equal capital_")
    _run(''.join(lines))


def send_drawdown_recovery_alert(report: dict):
    """
    Sends a Telegram alert when the recovery tier changes to
    REDUCED or below — tells you the system has auto-scaled down.
    """
    tier    = report.get('dd_tier', '')
    mult    = report.get('size_multiplier', 1.0) * 100
    dd_pct  = report.get('current_drawdown_pct', 0)
    streak  = report.get('consecutive_losses', 0)
    today   = date.today().strftime('%d %b %Y')

    emoji = {'NORMAL':'🟢','CAUTION':'🟡','REDUCED':'🟠',
             'DEFENSIVE':'🔴','CRITICAL':'🔴','HALTED':'🚫'}.get(tier, '⚪')

    msg = (
        f"{emoji} *QuantAI Recovery Alert* — {today}\n\n"
        f"Drawdown tier : *{tier}*\n"
        f"Position sizes: *{mult:.0f}%* of normal\n"
        f"Portfolio DD  : `{dd_pct:+.1f}%` from peak\n"
        f"Loss streak   : `{streak}` consecutive\n\n"
        f"_All new positions auto-scaled to {mult:.0f}% until recovery_"
    )
    _run(msg)
