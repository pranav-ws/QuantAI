import json
import pandas as pd # type: ignore
from datetime import datetime, date,timedelta
from src.paper_trader import (load_state, save_state,
                               get_signal, fetch_latest_data)
from src.risk import RiskManager
from src.data_collector import STOCK_UNIVERSE

# Set this to True once you've added your BOT_TOKEN and CHAT_ID
# in src/alerts.py to get BUY signals sent to Telegram.
ENABLE_TELEGRAM_ALERTS = True

def run_paper_trading_session():
    print(f"\n{'='*58}")
    print(f"  QuantAI Paper Trading — "
          f"{datetime.now().strftime('%d %b %Y, %I:%M %p')}")
    print(f"{'='*58}")

    # ── Drift check ───────────────────────────────────────
    try:
        from src.drift_detector import (
            update_outcomes, check_accuracy_drift,
            check_regime_drift
        )
        update_outcomes()
        drift, accuracy, _ = check_accuracy_drift()
        reg_drift, regime, _ = check_regime_drift()

        if accuracy:
            status = '🔴 DRIFT' if drift else '🟢 HEALTHY'
            print(f"\n  Model Health: {accuracy:.1%} {status}")

        if drift or reg_drift:
            print(f"  Triggering adaptive retrain...")
            from src.adaptive_trainer import \
                run_adaptive_retraining
            run_adaptive_retraining(
                max_retrain=3,
                reason='pre_scan_drift_check'
            )
    except Exception as e:
        print(f"  Drift check skipped: {e}")

    # ── Factor model summary ──────────────────────────────
    try:
        from src.factor_model import rank_all_stocks
        factor_ranks = rank_all_stocks(use_cache_hours=24)
        top_factor   = [r for r in factor_ranks
                        if r['tier'] == 'TOP'][:5]
        if top_factor:
            top_names = ', '.join([
                r['ticker'].replace('.NS', '')
                for r in top_factor
            ])
            print(f"\n  📊 Top Factor Stocks: {top_names}")
    except Exception:
        pass

    # ── Refresh sector rotation ───────────────────────────
    try:
        from src.sector_rotation import get_rotation_signals
        ranked, buy_s, avoid_s, best = \
            get_rotation_signals(top_n_sectors=3)
        top_sectors = [r['sector']
                       for r in ranked
                       if r['tier'] == 'TOP']
        print(f"\n  🔄 Top Sectors: "
              f"{' | '.join(top_sectors)}")
    except Exception as e:
        print(f"  ⚠️  Sector rotation skipped: {e}")


    # ── Tail risk check (Black Swan Detector) ────────────
    try:
        from src.tail_risk import TailRiskMonitor, LEVEL_NORMAL
        tri_monitor = TailRiskMonitor()
        tri_report  = tri_monitor.scan_from_db(period_days=252)
        tri_level   = tri_report.level
        tri_value   = tri_report.tri

        print(f"\n  {'─'*54}")
        print(f"  🔴 Tail Risk Monitor  :  {tri_level}  (TRI = {tri_value:.3f})")

        if tri_report.halt_trading:
            print(f"  🚫 BLACK SWAN ALERT — all trading suppressed")
            print(f"     {tri_report.recommendation}")
            for reason in tri_report.reasons:
                print(f"     {reason}")
            print(f"\n{'='*58}\n")
            return trade_log          # exit immediately — no trades today

        elif tri_level != LEVEL_NORMAL:
            print(f"  ⚠️  Elevated tail risk — reducing position sizes")
            for reason in tri_report.reasons:
                print(f"     {reason}")
            rm.apply_tail_risk_sizing(tri_level)

    except Exception as e:
        tri_report = None
        tri_level  = LEVEL_NORMAL
        print(f"  ⚠️  Tail risk check skipped: {e}")
    print(f"  {'─'*54}")

    # ── Detect market regime ──────────────────────────────
    try:
        from src.regime_detector import (detect_regime,
            get_regime_threshold, get_regime_summary, REGIMES)
        regime_data      = detect_regime()
        regime           = regime_data['regime']
        regime_threshold = get_regime_threshold(regime_data)
        regime_summary   = get_regime_summary(regime_data)
        max_positions    = regime_data['trading']['max_positions']
        print(f"\n  {regime_summary}")
        print(f"  Confidence threshold adjusted to "
              f"{regime_threshold:.0%} for {regime} regime")
        
    except Exception as e:
        regime           = 'SIDEWAYS'
        regime_threshold = 0.58
        max_positions    = 4
        print(f"  ⚠️  Regime detection skipped: {e}")

    # Load saved state
    cap_state, trade_log = load_state()
    capital = cap_state['capital']
    rm      = RiskManager(initial_capital=cap_state['peak'])
    rm.capital = capital

    # ── Drawdown recovery manager ─────────────────────────
    from src.drawdown_recovery import DrawdownRecoveryManager
    drm = DrawdownRecoveryManager(initial_capital=cap_state['peak'])
    drm.load_state()
    drm.update_capital(capital, cap_state['peak'])
    drm.print_report()

    print(f"\n  💰 Current Capital  : ₹{capital:>10,.0f}")
    print(f"  📅 Trading Since    : {cap_state['start']}")
    print(f"  📋 Total Trades     : {len(trade_log)}")


    # ── Close stop-loss-hit positions + update recovery ───
    open_trades = [t for t in trade_log if t.get('status') == 'OPEN']
    if open_trades:
        print(f"\n  🔍 Checking {len(open_trades)} open position(s)...")
        for t in open_trades:
            try:
                ticker = t['ticker']
                df_chk = fetch_latest_data(ticker)
                if df_chk is None:
                    continue
                current_price = float(df_chk.iloc[-1]['Close'])
                stop_loss     = t.get('stop_loss', 0)
                entry_price   = t.get('price', current_price)
                if stop_loss > 0 and current_price <= stop_loss:
                    pnl_inr = (current_price - entry_price) * t.get('shares', 0)
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    t['status']       = 'CLOSED'
                    t['exit_price']   = round(current_price, 2)
                    t['pnl']          = round(pnl_inr, 2)
                    t['exit_date']    = str(date.today())
                    capital          += pnl_inr
                    cap_state['capital'] = round(capital, 2)
                    cap_state['peak']    = max(cap_state['peak'], capital)
                    result = 'WIN' if pnl_inr > 0 else 'LOSS'
                    drm.update(result, pnl_pct, capital, cap_state['peak'])
                    print(f"  {'✅' if result=='WIN' else '❌'} CLOSED {ticker}  "
                          f"@ ₹{current_price:.1f}  P&L: ₹{pnl_inr:+,.0f}  ({pnl_pct:+.1f}%)")
            except Exception as e:
                print(f"  ⚠️  Could not check {t.get('ticker','?')}: {e}")

    print(f"\n  {'─'*54}")
    print(f"  {'TICKER':<20} {'PRICE':>8} {'CONF':>7} {'SIGNAL':>10}")
    print(f"  {'─'*54}")

    signals_today = []

    for ticker in STOCK_UNIVERSE:
        prediction, confidence, price, model_type = get_signal(ticker)

        if prediction is None:
            print(f"  {ticker:<20} {'N/A':>8} {'N/A':>7}  {'❓ NO MODEL'}")
            continue

        if price is None:
            print(f"  {ticker:<20} {'N/A':>8} {'N/A':>7}  {'❌ NO PRICE'}")
            continue

        confidence = 0.0 if confidence is None else confidence

        # ── Apply news sentiment modifier ────────────────
        try:
            from src.news_sentiment import get_sentiment, get_sentiment_modifier, sentiment_label
            sent_score, _, _ = get_sentiment(ticker, max_articles=5)
            modifier       = get_sentiment_modifier(sent_score)
            confidence_val = confidence
            adj_conf       = min(float(confidence_val) * modifier, 1.0)
            sent_str       = sentiment_label(sent_score)
        except Exception:
            confidence_val = confidence
            adj_conf = float(confidence_val)
            sent_str = ''
            modifier = 1.0

        # Re-evaluate signal with adjusted confidence
        prediction = 1 if adj_conf >= regime_threshold else 0

        if prediction == 1:
            signal_str = f'🟢 BUY [{model_type}]'
        else:
            signal_str = '🔴 SKIP'

        mod_str = f'({modifier:+.0%})' if modifier != 1.0 else ''
        print(f"  {ticker:<20} ₹{price:>7.1f} {adj_conf:>6.1%}{mod_str:<7}  {sent_str:<20} {signal_str}")

        if prediction == 1:
            shares, stop_loss, reason = rm.calculate_position_size(
                capital=capital,
                price=price,
                confidence=confidence
            )
            if shares > 0:
                # Apply drawdown recovery multiplier
                rec_mult = drm.state.size_multiplier
                if rec_mult < 1.0:
                    shares = max(1, int(shares * rec_mult))
                    print(f"  ⚖️  Recovery scaling: {rec_mult*100:.0f}% → {shares} shares")
                trade_value = shares * price
                # Respect regime max positions
                if len(signals_today) >= max_positions:
                    print(f"  ⏹  Max positions "
                          f"({max_positions}) reached "
                          f"for {regime} regime")
                    break
                signals_today.append({
                    'ticker'     : ticker,
                    'price'      : round(price, 2),
                    'confidence' : round(confidence, 4),
                    'shares'     : shares,
                    'stop_loss'  : round(stop_loss, 2),
                    'trade_value': round(trade_value, 2),
                    'date'       : str(date.today()),
                    'status'     : 'OPEN'
                })
    # ── Risk parity re-sizing ─────────────────────────────
    rp_result = None
    if signals_today:
        try:
            from src.risk_parity import RiskParityAllocator
            rp_allocator         = RiskParityAllocator()
            signals_today, rp_result = rp_allocator.allocate(
                signals_today, capital
            )
            rp_allocator.print_allocation(rp_result, signals_today)
        except Exception as e:
            print(f"  ⚠️  Risk parity skipped: {e}")

        
    # ── Summary of today's trades ────────────────────────
    print(f"\n  {'─'*54}")
    if signals_today:
        print(f"\n  📢 {len(signals_today)} TRADE SIGNAL(S) FOR TOMORROW:\n")
        total_deployed = 0
        for t in signals_today:
            print(f"  BUY  {t['ticker']}")
            print(f"       {t['shares']} shares @ ₹{t['price']:.1f}")
            print(f"       Value    : ₹{t['trade_value']:,.0f}")
            print(f"       Stop Loss: ₹{t['stop_loss']:.1f}")
            print(f"       Confidence: {t['confidence']:.1%}")
            if 'rp_weight' in t:
                print(f"       RP Weight : {t['rp_weight']*100:.1f}%  "
                      f"Daily Vol: {t['rp_daily_vol']:.2f}%/day")
            print()
            total_deployed += t['trade_value']

        pct_deployed = total_deployed / capital * 100
        print(f"  Total Capital Deployed: ₹{total_deployed:,.0f} ({pct_deployed:.1f}%)")
        print(f"  Cash Remaining        : ₹{capital - total_deployed:,.0f}")

        # Optional Telegram alert
        if ENABLE_TELEGRAM_ALERTS:
            try:
                from src.alerts import (send_signal_alert,
                                         send_regime_alert,
                                         send_tail_risk_alert,
                                         send_risk_parity_alert)
                send_regime_alert(regime_data)
                send_signal_alert(signals_today)
                from src.alerts import send_risk_parity_alert
                if 'rp_result' in dir():
                    send_risk_parity_alert(rp_result, signals_today)
                if tri_report and tri_level != LEVEL_NORMAL:
                    send_tail_risk_alert(tri_report)
                if rp_result is not None:
                    send_risk_parity_alert(rp_result, signals_today)
            except Exception as e:
                print(f"  ⚠️  Telegram alert failed: {e}") 
                # Weekly performance alert (Fridays only)
    if datetime.now().weekday() == 4:
        try:
            from src.performance_tracker import (
                calculate_metrics, load_all_trades
            )
            from src.alerts import send_weekly_summary
            all_trades = load_all_trades()
            send_weekly_summary(all_trades)
        except Exception:
            pass
    else:
        print(f"\n  💤 No high-confidence signals today — staying in cash.")
        print(f"     This is correct behaviour in uncertain markets.")

    # Save updated state
    trade_log.extend(signals_today)
    save_state(cap_state, trade_log)
    drm.save_state()

    # ── Auto-close yesterday's open trades ───────────────
    try:
        from src.performance_tracker import close_trade
        import sqlite3

        yesterday = (date.today() -
                     timedelta(days=1)).strftime('%Y-%m-%d')
        conn = sqlite3.connect('data/quantai.db')

        for t in trade_log:
            if (t.get('status') == 'OPEN' and
                    t.get('date', '') <= yesterday):
                ticker = t.get('ticker')
                result = pd.read_sql_query(
                    "SELECT close FROM prices "
                    "WHERE ticker=? AND date=? LIMIT 1",
                    conn, params=(ticker, str(date.today()))
                )
                if not result.empty:
                    exit_p = float(result['close'].iloc[0])
                    entry  = t.get('price', exit_p)
                    shares = t.get('shares', 0)
                    pnl    = (exit_p - entry) * shares
                    pnl_pct= (exit_p - entry)/entry*100

                    t['status']    = 'CLOSED'
                    t['exit_price']= round(exit_p, 2)
                    t['exit_date'] = str(date.today())
                    t['pnl']       = round(pnl, 2)
                    t['pnl_pct']   = round(pnl_pct, 2)
        conn.close()
    except Exception as e:
        print(f"  Auto-close skipped: {e}")

    
    # ── Trade history ────────────────────────────────────
    if trade_log:
        closed = [t for t in trade_log if t.get('status') == 'CLOSED']
        if closed:
            wins   = sum(1 for t in closed if t.get('pnl', 0) > 0)
            losses = sum(1 for t in closed if t.get('pnl', 0) <= 0)
            print(f"\n  📊 Paper Trade Record:")
            print(f"     Closed trades : {len(closed)}")
            print(f"     Wins / Losses : {wins} / {losses}")
            if len(closed) > 0:
                win_rate = wins / len(closed) * 100
                print(f"     Win Rate      : {win_rate:.1f}%")

    # ── VaR Report ────────────────────────────────────────
    try:
        from src.var_calculator import generate_var_report
        from src.alerts import send_var_alert
        var_report = generate_var_report(
            capital     = capital,
            open_trades = [t for t in trade_log
                           if t.get('status') == 'OPEN']
        )
        port_var = var_report.get('portfolio_var')
        if port_var:
            print(f"\n  📉 Portfolio VaR (95%): "
                  f"₹{port_var['conservative_var_inr']:,.0f} "
                  f"({port_var['conservative_var_pct']:.2f}%)")
            print(f"  🛡  Diversif. benefit : "
                  f"₹{port_var['diversification_benefit']:,.0f}")
        else:
            print(f"\n  📉 VaR: ₹0 — no open positions")
        if ENABLE_TELEGRAM_ALERTS:
            from src.alerts import send_drawdown_recovery_alert
            rec_report = drm.get_report()
            if rec_report['dd_tier'] not in ('NORMAL',) or \
                   rec_report['streak_tier'] not in ('NORMAL',):
                    send_drawdown_recovery_alert(rec_report)
            send_var_alert(var_report)
    except Exception as e:
        print(f"  ⚠️  VaR skipped: {e}")

    print(f"\n{'='*58}\n")
    return trade_log
if __name__ == '__main__':
    trade_log = run_paper_trading_session()

    # ── Trade history ────────────────────────────────────
    if trade_log:
        closed = [t for t in trade_log if t.get('status') == 'CLOSED']
        if closed:
            wins   = sum(1 for t in closed if t.get('pnl', 0) > 0)
            losses = sum(1 for t in closed if t.get('pnl', 0) <= 0)
            print(f"\n  📊 Paper Trade Record:")
            print(f"     Closed trades : {len(closed)}")
            print(f"     Wins / Losses : {wins} / {losses}")
            if len(closed) > 0:
                win_rate = wins / len(closed) * 100
                print(f"     Win Rate      : {win_rate:.1f}%")

    print(f"\n{'='*58}\n")
    
