"""
backtest_ensemble.py

Backtests the ensemble model on one stock and compares three columns:
  1. ML-only Ensemble   — RF + XGBoost + SeqNN (original)
  2. Full Ensemble      — ML + rule-based strategies (new, when they fire)
  3. Buy & Hold         — passive baseline

The rule-based strategies (Mean Reversion, Momentum Breakout,
MACD+RSI Confluence) only enter the vote on days they fire a
tradeable signal — on all other days the ensemble falls back to
pure ML, so "Full Ensemble" ≥ "ML-only" in terms of information.

The comparison answers the question: do the rule-based votes improve
signal quality, add noise, or are they mostly silent (not firing)?
"""
import os, joblib # type: ignore
import pandas as pd # type: ignore
import numpy as np # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt # type: ignore
import matplotlib.gridspec as gridspec# type: ignore
from src.features import get_feature_dataset
from src.model import prepare_data, time_series_split, FEATURE_COLS
from src.ensemble_model import (get_ensemble_confidence,
                                 get_ensemble_signal_full,
                                 get_model_agreement,
                                 _get_rule_based_confidences,
                                 MODEL_WEIGHTS)

TICKER          = 'RELIANCE.NS'
INITIAL_CAPITAL = 100_000
THRESHOLD       = 0.58


def run_backtest_for_signals(df, signals, capital=INITIAL_CAPITAL):
    """
    Generic next-bar backtest given a list of 0/1 signals aligned to df.
    Position is held for exactly one bar then re-evaluated — same
    logic as the original backtest_ensemble.py.
    """
    capital     = float(capital)
    position    = 0
    entry_price = 0.0
    equity      = []
    trades      = []

    for i, (date, row) in enumerate(df.iterrows()):
        price = row['Close']
        if position > 0:
            pnl     = (price - entry_price) * position
            capital += price * position
            trades.append({'pnl': pnl, 'result': 'WIN' if pnl > 0 else 'LOSS'})
            position = 0
        if i < len(signals) and signals[i] == 1 and position == 0:
            position    = int(capital * 0.95 / price)
            entry_price = price
            capital    -= position * price
        equity.append(capital + position * price)

    eq       = pd.Series(equity, index=df.index)
    total_ret = (eq.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    bh_ret    = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
    daily_r   = eq.pct_change().dropna()
    sharpe    = (daily_r.mean() / daily_r.std()) * np.sqrt(252) if daily_r.std() > 0 else 0
    max_dd    = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    wins      = sum(1 for t in trades if t['result'] == 'WIN')
    win_rate  = wins / len(trades) * 100 if trades else 0

    return eq, {
        'total_return': total_ret, 'bh_return': bh_ret,
        'sharpe': sharpe, 'max_dd': max_dd,
        'n_trades': len(trades), 'win_rate': win_rate
    }


# ── Load data ─────────────────────────────────────────────
print(f"\n⚙️  Loading data for {TICKER}...")
df_full  = get_feature_dataset(TICKER)
X, y     = prepare_data(df_full)
_, X_test, _, y_test = time_series_split(X, y, test_size=0.2)
test_df  = df_full.loc[X_test.index]

# ── Generate signals — ML-only and Full (ML + rule-based) ──
print("🎯 Generating signals (ML-only and full ensemble)...")
ml_only_signals  = []
full_ens_signals = []
rb_fire_log      = []   # track when / which rule-based strategies voted

for i, (date, row) in enumerate(test_df.iterrows()):
    hist_df = df_full.loc[:date]

    # ML-only: use get_ensemble_confidence, which includes rule-based
    # strategies too — so to get truly ML-only we need to compute
    # ML models independently.
    ml_conf = None
    try:
        from src.ensemble_model import _load_models
        ml_models = _load_models(TICKER)
        ml_confs  = {}
        latest    = hist_df.iloc[-1]
        from src.lstm_model import predict_lstm, SEQUENCE_LEN
        for name, model_obj in ml_models.items():
            if name in ('SeqNN', 'LSTM'):
                if len(hist_df) >= SEQUENCE_LEN:
                    ml_confs[name] = predict_lstm(model_obj[0], model_obj[1], hist_df)
            else:
                X_row = pd.DataFrame([latest[FEATURE_COLS]], columns=FEATURE_COLS)
                ml_confs[name] = float(model_obj.predict_proba(X_row)[0][1])
        if ml_confs:
            total_w = sum(MODEL_WEIGHTS.get(k, 1.0) for k in ml_confs)
            ml_conf = sum(ml_confs[k] * MODEL_WEIGHTS.get(k, 1.0) for k in ml_confs) / total_w
    except Exception:
        pass

    # Full ensemble: ML + any rule-based that fired today
    full_conf, individual, _, rb_details, ml_count, rb_count = \
        get_ensemble_signal_full(TICKER, hist_df)

    ml_only_signals.append(1 if (ml_conf or 0) >= THRESHOLD else 0)
    full_ens_signals.append(1 if (full_conf or 0) >= THRESHOLD else 0)

    if rb_count > 0:
        rb_fire_log.append({
            'date'         : date,
            'rb_strategies': list(rb_details.keys()),
            'rb_signals'   : {k: v['signal']     for k, v in rb_details.items()},
            'rb_confs'     : {k: v['confidence'] for k, v in rb_details.items()},
            'ml_conf'      : round(ml_conf, 4) if ml_conf else None,
            'full_conf'    : round(full_conf, 4) if full_conf else None,
            'changed_signal': (
                (1 if (ml_conf or 0) >= THRESHOLD else 0) !=
                (1 if (full_conf or 0) >= THRESHOLD else 0)
            ),
        })

# ── Best individual ML model signals ─────────────────────
print("📊 Generating best individual model signals...")
best_model = None
best_path  = None
for suffix in ['xgb_model.pkl', 'rf_model.pkl']:
    p = os.path.join('models', f'{TICKER}_{suffix}')
    if os.path.exists(p):
        best_model = joblib.load(p)
        best_path  = suffix
        break

ind_signals = []
if best_model:
    proba = best_model.predict_proba(X_test)[:, 1]
    ind_signals = (proba >= THRESHOLD).astype(int).tolist()

# ── Run backtests ─────────────────────────────────────────
eq_ml,   m_ml   = run_backtest_for_signals(test_df, ml_only_signals)
eq_full, m_full = run_backtest_for_signals(test_df.copy(), full_ens_signals)
eq_ind,  m_ind  = (run_backtest_for_signals(test_df.copy(), ind_signals)
                   if ind_signals else (None, None))

# ── Print comparison ──────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Backtest Results — {TICKER}")
print(f"{'='*70}")
print(f"  {'Metric':<26} {'Full Ens':>12} {'ML-only':>12} {'Individual':>12}")
print(f"  {'-'*66}")

metrics = [
    ('Total Return',  'total_return', '%'),
    ('Buy & Hold',    'bh_return',    '%'),
    ('Sharpe Ratio',  'sharpe',       ''),
    ('Max Drawdown',  'max_dd',       '%'),
    ('Trades',        'n_trades',     ''),
    ('Win Rate',      'win_rate',     '%'),
]
for label, key, unit in metrics:
    f_val = m_full.get(key, 0)
    m_val = m_ml.get(key, 0)
    i_val = m_ind.get(key, 0) if m_ind else 0

    fmt = lambda v, u: f'{v:+.2f}{u}' if u else f'{v:.2f}'
    f_str = fmt(f_val, unit)
    m_str = fmt(m_val, unit)
    i_str = fmt(i_val, unit) if m_ind else 'N/A'

    best  = '✅' if key not in ('max_dd', 'n_trades') and f_val >= m_val \
            else ('✅' if key == 'max_dd' and f_val >= m_val else '  ')
    print(f"  {best} {label:<24} {f_str:>12} {m_str:>12} {i_str:>12}")

print(f"{'='*70}")

# ── Rule-based voting summary ─────────────────────────────
rb_df = pd.DataFrame(rb_fire_log) if rb_fire_log else None
print(f"\n  📋 RULE-BASED STRATEGY VOTE SUMMARY")
print(f"  {'─'*50}")
if rb_df is None or rb_df.empty:
    print("  ℹ️  No rule-based strategies fired during the test period.")
    print("  (This is expected — confluence conditions are rare by design.)")
else:
    total_rb_days   = len(rb_df)
    signal_changed  = rb_df['changed_signal'].sum()
    print(f"  Days a rule-based strategy voted:   {total_rb_days}")
    print(f"  Days that CHANGED the final signal: {signal_changed}")
    print(f"\n  Strategy-level fire counts (out of {len(test_df)} test days):")
    all_rb_names = ['MeanReversion', 'MomentumBreakout', 'MACD_RSI_Confluence']
    for name in all_rb_names:
        count = sum(1 for r in rb_fire_log if name in r['rb_strategies'])
        print(f"    {name:<26}  {count:>4} days")
    if signal_changed > 0:
        changed = rb_df[rb_df['changed_signal']]
        print(f"\n  Days the rule-based vote flipped the signal:")
        for _, r in changed.iterrows():
            print(f"    {str(r['date'])[:10]}  ML:{r['ml_conf']:.3f} → Full:{r['full_conf']:.3f}  "
                  f"strategies: {', '.join(r['rb_strategies'])}")
print(f"\n{'='*70}\n")

# ── Plot ──────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor='#0d0d1a')
gs  = fig.add_gridspec(2, 1, height_ratios=[2.5, 1], hspace=0.3)

ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor('#1a1a2e')

eq_full_norm = eq_full / INITIAL_CAPITAL * 100
eq_ml_norm   = eq_ml   / INITIAL_CAPITAL * 100
price_norm   = test_df['Close'] / test_df['Close'].iloc[0] * 100

ax1.plot(eq_full.index, eq_full_norm, color='#fbbf24', linewidth=2.0,
         label='Full Ensemble (ML + Rule-based)', zorder=4)
ax1.plot(eq_ml.index, eq_ml_norm, color='#4ecdc4', linewidth=1.5,
         linestyle='--', label='ML-only Ensemble', alpha=0.85, zorder=3)
if eq_ind is not None:
    eq_ind_norm = eq_ind / INITIAL_CAPITAL * 100
    ax1.plot(eq_ind.index, eq_ind_norm, color='#a78bfa', linewidth=1.2,
             linestyle=':', label=f'Best Individual ({best_path or ""})', alpha=0.8)
ax1.plot(test_df.index, price_norm, color='#ff6b6b', linewidth=1.1,
         linestyle=':', label='Buy & Hold', alpha=0.7)

# Mark rule-based vote days on the chart
if rb_df is not None and not rb_df.empty:
    rb_dates = pd.to_datetime(rb_df['date'])
    rb_vals  = eq_full_norm.reindex(rb_dates, method='nearest')
    ax1.scatter(rb_dates, rb_vals, marker='D', color='#f97316', s=35,
                zorder=5, label='Rule-based vote fired', alpha=0.8)

ax1.axhline(100, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
ax1.fill_between(eq_full.index, eq_full_norm, 100,
                 where=(eq_full_norm >= 100), color='#fbbf24', alpha=0.06)
ax1.fill_between(eq_full.index, eq_full_norm, 100,
                 where=(eq_full_norm <  100), color='red',     alpha=0.06)
ax1.set_title(f'Full Ensemble vs ML-only vs Buy & Hold — {TICKER}',
              color='white', fontsize=12, pad=12)
ax1.set_ylabel('Portfolio Value (base 100)', color='white')
ax1.legend(fontsize=9, facecolor='#12121f', labelcolor='white', edgecolor='#262640')
ax1.tick_params(colors='white')

ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor('#1a1a2e')
for eq_curve, color, label in [
    (eq_full, '#fbbf24', 'Full Ensemble'),
    (eq_ml,   '#4ecdc4', 'ML-only'),
]:
    rolling_max = eq_curve.cummax()
    drawdown    = (eq_curve - rolling_max) / rolling_max * 100
    ax2.plot(eq_curve.index, drawdown, color=color, linewidth=0.9, label=label)
    if color == '#fbbf24':
        ax2.fill_between(eq_curve.index, drawdown, 0, color=color, alpha=0.15)
ax2.axhline(-20, color='orange', linestyle='--', linewidth=0.8, alpha=0.7, label='-20% limit')
ax2.set_ylabel('Drawdown %', color='white', fontsize=9)
ax2.set_title('Drawdown Comparison', color='white', fontsize=10)
ax2.legend(fontsize=8, facecolor='#12121f', labelcolor='white', edgecolor='#262640')
ax2.tick_params(colors='white')

os.makedirs('models', exist_ok=True)
plt.savefig('models/ensemble_backtest.png', dpi=150, bbox_inches='tight', facecolor='#0d0d1a')
plt.show()
print("📊 Chart saved → models/ensemble_backtest.png")