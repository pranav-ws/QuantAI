"""
src/alpha_attribution.py

Alpha Attribution — Where Does Our Edge Come From?

Standard performance reports answer "how much did we make?"
This module answers "WHY did we make it?" — decomposing total alpha
(outperformance vs a Nifty buy-and-hold benchmark) into its sources.

Six attribution layers, each independently computable:

  1. FEATURE ATTRIBUTION
     Which technical indicators drove profitable signals?
     Method: For each feature, compute (importance × PnL-correlation):
     - importance  = model's feature_importances_ (RF/XGBoost)
     - PnL-corr    = Pearson correlation between the feature's value
                     at trade entry and the resulting trade PnL
     Combined: high importance AND high PnL-correlation = genuine
     alpha driver. High importance but zero PnL-correlation = the
     model thinks it matters but it doesn't translate to real P&L.
     Results are grouped into 5 signal families (Trend, Momentum,
     Volatility, Volume, Price Action).

  2. SECTOR ATTRIBUTION (Brinson-Hood-Beebower decomposition)
     Did we pick the right sectors, and within them the right stocks?
     - Allocation effect: over/underweighting sectors vs benchmark
     - Selection effect:  our stock picks within a sector vs the
                          sector's average return
     - Total attribution = allocation + selection

  3. CONFIDENCE ATTRIBUTION
     Does a higher model confidence actually mean better trades?
     Buckets: 58-62%, 62-66%, 66-70%, 70-75%, 75%+
     Each bucket shows: trade count, win rate, avg P&L.
     If confidence is uncorrelated with P&L, the model's calibration
     needs work. If highly correlated, confidence is a reliable filter.

  4. TIMING ATTRIBUTION
     Is alpha concentrated in a few lucky months or spread evenly?
     Monthly P&L breakdown shows whether the strategy is robust
     across different market conditions or dependent on one good run.

  5. RULE-BASED STRATEGY ATTRIBUTION
     When MeanReversion / MomentumBreakout / MACD+RSI Confluence
     fired alongside the ML ensemble, did those trades outperform
     trades where only the ML model voted?
     Requires trades to have been tagged with 'rule_based_voted': bool.
     Trades without this tag are treated as ML-only.

  6. MODEL ATTRIBUTION
     Which model within the ensemble contributed most alpha?
     Uses per-model predictions stored in individual confidence scores.
     Requires trades to have 'individual_confs' dict logged at entry.
"""

import os
import json
import sqlite3
import joblib# type: ignore
import numpy as np# type: ignore
import pandas as pd# type: ignore
from dataclasses import dataclass, field
from typing import Optional


# ── Feature group taxonomy ────────────────────────────────

FEATURE_GROUPS = {
    # Trend-following indicators
    'SMA_20'        : 'Trend',
    'SMA_50'        : 'Trend',
    'EMA_9'         : 'Trend',
    'EMA_21'        : 'Trend',
    'Close_vs_SMA20': 'Trend',
    'Close_vs_SMA50': 'Trend',

    # Momentum oscillators
    'RSI_14'        : 'Momentum',
    'MACD'          : 'Momentum',
    'MACD_Signal'   : 'Momentum',
    'MACD_Hist'     : 'Momentum',
    'Stoch_K'       : 'Momentum',
    'Stoch_D'       : 'Momentum',

    # Volatility regime
    'BB_Upper'      : 'Volatility',
    'BB_Middle'     : 'Volatility',
    'BB_Lower'      : 'Volatility',
    'BB_Width'      : 'Volatility',
    'ATR_14'        : 'Volatility',

    # Volume flow
    'OBV'           : 'Volume',
    'Volume_Ratio'  : 'Volume',

    # Price action
    'Daily_Return'  : 'Price Action',
    'Return_5d'     : 'Price Action',
    'Return_20d'    : 'Price Action',
    'High_Low_Range': 'Price Action',
}

GROUP_COLORS = {
    'Trend'       : '#4ecdc4',
    'Momentum'    : '#fbbf24',
    'Volatility'  : '#a78bfa',
    'Volume'      : '#f97316',
    'Price Action': '#22c55e',
}

# Confidence buckets (lower bound, upper bound, label)
CONFIDENCE_BUCKETS = [
    (0.58, 0.62, '58–62%'),
    (0.62, 0.66, '62–66%'),
    (0.66, 0.70, '66–70%'),
    (0.70, 0.75, '70–75%'),
    (0.75, 1.01, '75%+'),
]


# ── Report dataclass ──────────────────────────────────────

@dataclass
class AlphaReport:
    """Complete attribution report. All fields have sensible defaults
    so partial attribution (e.g. no trade log yet) still works."""

    ticker              : str   = ''
    total_return_pct    : float = 0.0
    benchmark_return_pct: float = 0.0
    total_alpha_pct     : float = 0.0
    n_trades            : int   = 0
    win_rate            : float = 0.0
    sharpe              : float = 0.0

    # Layer 1 — Feature
    feature_contributions : dict  = field(default_factory=dict)   # {feature: alpha_contrib}
    group_contributions   : dict  = field(default_factory=dict)   # {group: total_contrib}
    top_features          : list  = field(default_factory=list)   # [(feature, contrib), ...]

    # Layer 2 — Sector
    sector_attribution    : dict  = field(default_factory=dict)
    # {sector: {n_trades, win_rate, total_pnl_pct, selection_effect, allocation_effect}}

    # Layer 3 — Confidence
    confidence_buckets    : dict  = field(default_factory=dict)
    # {label: {n_trades, win_rate, avg_pnl_pct, total_pnl_pct}}

    # Layer 4 — Timing
    monthly_pnl           : dict  = field(default_factory=dict)   # {YYYY-MM: pnl_pct}
    best_month            : str   = ''
    worst_month           : str   = ''

    # Layer 5 — Rule-based
    rule_based_attribution: dict  = field(default_factory=dict)
    # {strategy: {n_trades, avg_pnl_with, avg_pnl_without, delta}}

    # Layer 6 — Model
    model_attribution     : dict  = field(default_factory=dict)
    # {model_name: {importance_share, pnl_contribution}}


# ── Core attributor ───────────────────────────────────────

class AlphaAttributor:

    def __init__(self):
        self._db_path = os.path.join('data', 'quantai.db')

    # ── Layer 1: Feature attribution ─────────────────────

    def attribute_features(self,
                            model,
                            X_test   : pd.DataFrame,
                            trade_pnls: pd.Series) -> dict:
        """
        Computes per-feature alpha contribution.

        For each feature:
          contribution = feature_importance × pearson_corr(feature_values, trade_pnl)

        Only rows where a trade was entered are used — feature values
        on "no-trade" days don't contribute to alpha.

        Returns {feature_name: contribution_score} — positive means this
        feature, when activated, led to positive PnL; negative means it
        was a detractor; zero means it had no predictive power for P&L.
        """
        if not hasattr(model, 'feature_importances_'):
            return {}

        from src.model import FEATURE_COLS
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS)

        # Align features with actual trade entries
        # trade_pnls index = dates where we entered a trade
        common_dates = X_test.index.intersection(trade_pnls.index)
        if len(common_dates) < 3:
            return {}

        X_trades  = X_test.loc[common_dates]
        pnl_trades = trade_pnls.loc[common_dates]

        contributions = {}
        for feat in FEATURE_COLS:
            if feat not in X_trades.columns:
                continue
            feat_vals = X_trades[feat].fillna(0)
            if feat_vals.std() == 0:
                contributions[feat] = 0.0
                continue
            # Pearson correlation of feature value with trade PnL
            corr = float(np.corrcoef(feat_vals.values,
                                      pnl_trades.values)[0, 1])
            if np.isnan(corr):
                corr = 0.0
            importance = float(importances.get(feat, 0.0))
            contributions[feat] = round(importance * corr, 5)

        return contributions

    def aggregate_groups(self, feature_contributions: dict) -> dict:
        """Sums feature contributions up to their group level."""
        groups = {}
        for feat, contrib in feature_contributions.items():
            grp = FEATURE_GROUPS.get(feat, 'Other')
            groups[grp] = groups.get(grp, 0.0) + contrib
        return {k: round(v, 5) for k, v in
                sorted(groups.items(), key=lambda x: abs(x[1]), reverse=True)}

    # ── Layer 2: Sector attribution (BHB) ────────────────

    def attribute_sectors(self,
                           trade_log  : list,
                           stock_universe: dict,
                           benchmark_sector_returns: Optional[dict] = None
                           ) -> dict:
        """
        Brinson-Hood-Beebower decomposition at sector level.

        Selection effect = (our avg PnL in sector) − (benchmark return in sector)
        Allocation effect = (our weight in sector − benchmark weight) × benchmark return
        Total = selection + allocation

        If benchmark_sector_returns is None, uses 0% as benchmark (absolute
        attribution — tells you which sectors made money in absolute terms).
        """
        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED' and 'pnl' in t]
        if not closed:
            return {}

        total_trades = len(closed)
        benchmark_weight = 1.0 / 16   # equal-weight across 16 sectors

        sector_data = {}
        for t in closed:
            ticker = t.get('ticker', '')
            sector = stock_universe.get(ticker, ('', 'Unknown'))[1]
            pnl_pct = t.get('pnl', 0) / (t.get('trade_value', 1) or 1) * 100

            if sector not in sector_data:
                sector_data[sector] = {'pnls': [], 'n': 0}
            sector_data[sector]['pnls'].append(pnl_pct)
            sector_data[sector]['n'] += 1

        result = {}
        for sector, data in sector_data.items():
            n      = data['n']
            pnls   = data['pnls']
            w      = n / total_trades
            avg_pnl = float(np.mean(pnls))
            wins   = sum(1 for p in pnls if p > 0)
            bm_ret  = (benchmark_sector_returns or {}).get(sector, 0.0)

            selection = avg_pnl - bm_ret
            allocation = (w - benchmark_weight) * bm_ret
            total_effect = selection + allocation

            result[sector] = {
                'n_trades'        : n,
                'weight_pct'      : round(w * 100, 1),
                'win_rate'        : round(wins / n * 100, 1) if n else 0,
                'avg_pnl_pct'     : round(avg_pnl, 2),
                'total_pnl_pct'   : round(sum(pnls), 2),
                'selection_effect': round(selection, 3),
                'allocation_effect': round(allocation, 3),
                'total_effect'    : round(total_effect, 3),
            }

        return dict(sorted(result.items(),
                           key=lambda x: x[1]['total_pnl_pct'], reverse=True))

    # ── Layer 3: Confidence attribution ──────────────────

    def attribute_confidence(self, trade_log: list) -> dict:
        """
        Groups trades by confidence bucket and shows performance per bucket.
        Answers: "Do high-confidence trades actually perform better?"
        """
        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED' and 'pnl' in t
                  and 'confidence' in t]
        if not closed:
            return {}

        result = {}
        for lo, hi, label in CONFIDENCE_BUCKETS:
            bucket = [t for t in closed
                      if lo <= t['confidence'] < hi]
            if not bucket:
                continue
            pnls  = [t['pnl'] / (t.get('trade_value', 1) or 1) * 100
                     for t in bucket]
            wins  = sum(1 for p in pnls if p > 0)
            result[label] = {
                'n_trades'     : len(bucket),
                'win_rate'     : round(wins / len(bucket) * 100, 1),
                'avg_pnl_pct'  : round(float(np.mean(pnls)), 2),
                'total_pnl_pct': round(float(np.sum(pnls)), 2),
                'conf_lo'      : lo,
                'conf_hi'      : hi,
            }

        return result

    # ── Layer 4: Timing attribution ───────────────────────

    def attribute_timing(self, trade_log: list) -> dict:
        """
        Monthly P&L breakdown.
        Shows if alpha is spread evenly or concentrated in lucky months.
        """
        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED' and 'pnl' in t]
        if not closed:
            return {}

        monthly = {}
        for t in closed:
            # Use exit_date if available, else date
            dt_str = t.get('exit_date') or t.get('date', '')
            if not dt_str:
                continue
            month = str(dt_str)[:7]   # YYYY-MM
            pnl_pct = t['pnl'] / (t.get('trade_value', 1) or 1) * 100
            if month not in monthly:
                monthly[month] = {'pnls': [], 'n': 0}
            monthly[month]['pnls'].append(pnl_pct)
            monthly[month]['n'] += 1

        result = {}
        for month, data in sorted(monthly.items()):
            pnls = data['pnls']
            wins = sum(1 for p in pnls if p > 0)
            result[month] = {
                'n_trades'     : data['n'],
                'win_rate'     : round(wins / data['n'] * 100, 1),
                'avg_pnl_pct'  : round(float(np.mean(pnls)), 2),
                'total_pnl_pct': round(float(np.sum(pnls)), 2),
            }

        return result

    # ── Layer 5: Rule-based strategy attribution ──────────

    def attribute_rule_based(self, trade_log: list) -> dict:
        """
        Compares trades where a rule-based strategy co-voted vs ML-only.
        Requires 'rule_based_voted' bool and optionally 'rule_based_strategies'
        list to be set on each trade dict (paper_trade.py adds these when
        ensemble_model.get_ensemble_signal_full() returns rb_count > 0).

        If trades don't have these fields, returns empty dict.
        """
        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED' and 'pnl' in t]
        if not closed:
            return {}

        rb_trades   = [t for t in closed if t.get('rule_based_voted', False)]
        ml_only     = [t for t in closed if not t.get('rule_based_voted', False)]

        if not rb_trades:
            return {}

        def _stats(trades):
            if not trades:
                return {'n': 0, 'avg_pnl': 0, 'win_rate': 0}
            pnls = [t['pnl'] / (t.get('trade_value', 1) or 1) * 100
                    for t in trades]
            wins = sum(1 for p in pnls if p > 0)
            return {
                'n'       : len(trades),
                'avg_pnl' : round(float(np.mean(pnls)), 2),
                'win_rate': round(wins / len(trades) * 100, 1),
            }

        overall = {
            'ML_only'      : _stats(ml_only),
            'With_RuleBase': _stats(rb_trades),
        }
        overall['With_RuleBase']['delta_avg_pnl'] = round(
            overall['With_RuleBase']['avg_pnl'] - overall['ML_only']['avg_pnl'], 2
        )

        # Per-strategy breakdown
        rb_strat_names = ['MeanReversion', 'MomentumBreakout', 'MACD_RSI_Confluence']
        for strat in rb_strat_names:
            strat_trades = [t for t in rb_trades
                            if strat in t.get('rule_based_strategies', [])]
            if strat_trades:
                s = _stats(strat_trades)
                s['delta_avg_pnl'] = round(s['avg_pnl'] - overall['ML_only']['avg_pnl'], 2)
                overall[strat] = s

        return overall

    # ── Layer 6: Model-level attribution ─────────────────

    def attribute_models(self, trade_log: list) -> dict:
        """
        When individual model confidences are logged per trade
        (via 'individual_confs' dict), this computes which model's
        signal had the highest correlation with actual trade PnL.
        Requires paper_trade.py to log individual_confs per trade.
        """
        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED'
                  and 'pnl' in t
                  and 'individual_confs' in t]
        if not closed:
            return {}

        model_names = set()
        for t in closed:
            model_names.update(t['individual_confs'].keys())

        result = {}
        for model in model_names:
            confs = []
            pnls  = []
            for t in closed:
                if model in t.get('individual_confs', {}):
                    confs.append(t['individual_confs'][model])
                    pnls.append(t['pnl'] / (t.get('trade_value', 1) or 1) * 100)
            if len(confs) < 3:
                continue
            corr = float(np.corrcoef(confs, pnls)[0, 1]) if np.std(confs) > 0 else 0.0
            if np.isnan(corr):
                corr = 0.0
            result[model] = {
                'n_trades'     : len(confs),
                'avg_conf'     : round(float(np.mean(confs)), 3),
                'pnl_corr'     : round(corr, 3),
                'avg_pnl_when_high_conf': round(
                    float(np.mean([p for c, p in zip(confs, pnls) if c >= 0.65])), 2
                ) if any(c >= 0.65 for c in confs) else 0.0,
            }

        return dict(sorted(result.items(),
                           key=lambda x: abs(x[1]['pnl_corr']), reverse=True))

    # ── Backtest-based attribution (single ticker) ────────

    def run_backtest_attribution(self,
                                  ticker          : str,
                                  confidence_threshold: float = 0.58
                                  ) -> AlphaReport:
        """
        Runs the full attribution on BACKTEST data for a single ticker.
        Uses the test split (last 20% of data) — same as src/backtest.py.
        This gives rich data for feature attribution because we have
        feature values at every trade entry, not just the live log.
        """
        from src.features import get_feature_dataset
        from src.model import prepare_data, time_series_split, FEATURE_COLS
        from src.data_collector import STOCK_UNIVERSE

        report        = AlphaReport(ticker=ticker)
        model_path    = os.path.join('models', f'{ticker}_rf_model.pkl')
        xgb_model_path= os.path.join('models', f'{ticker}_xgb_model.pkl')

        # Load best available model
        model = None
        for path in [xgb_model_path, model_path]:
            if os.path.exists(path):
                model = joblib.load(path)
                break

        if model is None:
            report.total_alpha_pct = 0.0
            return report

        df           = get_feature_dataset(ticker)
        X, y         = prepare_data(df)
        _, X_test, _, y_test = time_series_split(X, y, test_size=0.2)
        test_df      = df.loc[X_test.index]

        probas       = model.predict_proba(X_test)[:, 1]
        predictions  = (probas >= confidence_threshold).astype(int)
        confidences  = pd.Series(probas, index=X_test.index)

        # Simulate trades to get PnL per entry date
        entry_dates  = []
        entry_pnls   = []
        entry_confs  = []
        trade_log_bt = []
        position     = 0
        entry_price  = 0.0
        entry_date   = None

        capital = 100_000.0
        equity  = []

        for i, (date, row) in enumerate(test_df.iterrows()):
            price = row['Close']
            pred  = predictions[i]
            conf  = float(confidences.iloc[i])

            if position > 0:
                pnl_pct = (price - entry_price) / entry_price * 100
                pnl_inr = (price - entry_price) * position
                capital += price * position
                trade_log_bt.append({
                    'ticker'     : ticker,
                    'date'       : str(entry_date),
                    'exit_date'  : str(date),
                    'price'      : entry_price,
                    'exit_price' : price,
                    'confidence' : conf,
                    'trade_value': entry_price * position,
                    'pnl'        : pnl_inr,
                    'status'     : 'CLOSED',
                    'sector'     : STOCK_UNIVERSE.get(ticker, ('','Unknown'))[1],
                })
                entry_dates.append(entry_date)
                entry_pnls.append(pnl_pct)
                entry_confs.append(entry_conf)
                position = 0

            if pred == 1 and position == 0:
                position    = int(capital * 0.95 / price)
                entry_price = price
                entry_date  = date
                entry_conf  = conf
                capital    -= position * price

            equity.append(capital + position * price)

        if not entry_dates:
            return report

        # ── Compute headline metrics ──────────────────────
        equity_s     = pd.Series(equity, index=test_df.index)
        total_return = (equity_s.iloc[-1] - 100_000) / 100_000 * 100
        bh_return    = (test_df['Close'].iloc[-1] - test_df['Close'].iloc[0]) \
                       / test_df['Close'].iloc[0] * 100
        daily_r      = equity_s.pct_change().dropna()
        sharpe       = float((daily_r.mean() / daily_r.std()) * np.sqrt(252)) \
                       if daily_r.std() > 0 else 0.0
        wins         = sum(1 for p in entry_pnls if p > 0)

        report.total_return_pct     = round(total_return, 2)
        report.benchmark_return_pct = round(bh_return, 2)
        report.total_alpha_pct      = round(total_return - bh_return, 2)
        report.n_trades             = len(entry_pnls)
        report.win_rate             = round(wins / len(entry_pnls) * 100, 1)
        report.sharpe               = round(sharpe, 3)

        # ── Layer 1: Feature attribution ─────────────────
        trade_pnl_s  = pd.Series(entry_pnls, index=entry_dates)
        feat_contrib = self.attribute_features(model, X_test, trade_pnl_s)
        report.feature_contributions = feat_contrib
        report.group_contributions   = self.aggregate_groups(feat_contrib)
        report.top_features          = sorted(
            feat_contrib.items(), key=lambda x: abs(x[1]), reverse=True
        )[:10]

        # ── Layer 3: Confidence attribution ──────────────
        report.confidence_buckets = self.attribute_confidence(trade_log_bt)

        # ── Layer 4: Timing attribution ───────────────────
        report.monthly_pnl = self.attribute_timing(trade_log_bt)
        if report.monthly_pnl:
            best  = max(report.monthly_pnl, key=lambda m: report.monthly_pnl[m]['total_pnl_pct'])
            worst = min(report.monthly_pnl, key=lambda m: report.monthly_pnl[m]['total_pnl_pct'])
            report.best_month  = best
            report.worst_month = worst

        return report

    # ── Portfolio-level attribution (trade log) ───────────

    def run_portfolio_attribution(self, trade_log: list) -> AlphaReport:
        """
        Full attribution across the complete paper trading history.
        Runs all 6 layers on the live trade log.
        """
        from src.data_collector import STOCK_UNIVERSE

        report = AlphaReport(ticker='PORTFOLIO')

        closed = [t for t in trade_log
                  if t.get('status') == 'CLOSED' and 'pnl' in t]
        if not closed:
            return report

        all_pnls     = [t['pnl'] / (t.get('trade_value', 1) or 1) * 100
                        for t in closed]
        wins         = sum(1 for p in all_pnls if p > 0)
        total_pnl_pct = sum(all_pnls)

        report.n_trades   = len(closed)
        report.win_rate   = round(wins / len(closed) * 100, 1)
        report.total_return_pct = round(total_pnl_pct, 2)

        report.sector_attribution   = self.attribute_sectors(trade_log, STOCK_UNIVERSE)
        report.confidence_buckets   = self.attribute_confidence(trade_log)
        report.monthly_pnl          = self.attribute_timing(trade_log)
        report.rule_based_attribution = self.attribute_rule_based(trade_log)
        report.model_attribution    = self.attribute_models(trade_log)

        if report.monthly_pnl:
            report.best_month  = max(report.monthly_pnl,
                                      key=lambda m: report.monthly_pnl[m]['total_pnl_pct'])
            report.worst_month = min(report.monthly_pnl,
                                      key=lambda m: report.monthly_pnl[m]['total_pnl_pct'])

        return report

    # ── Print summary ─────────────────────────────────────

    def print_report(self, report: AlphaReport):
        ticker = report.ticker
        print(f"\n{'='*62}")
        print(f"  QuantAI Alpha Attribution — {ticker}")
        print(f"{'='*62}")
        print(f"  Total Return   : {report.total_return_pct:>+8.2f}%")
        if report.benchmark_return_pct:
            print(f"  Benchmark (B&H): {report.benchmark_return_pct:>+8.2f}%")
            print(f"  Alpha          : {report.total_alpha_pct:>+8.2f}%  ← the number we're decomposing")
        print(f"  Sharpe         : {report.sharpe:>8.3f}")
        print(f"  Win Rate       : {report.win_rate:>8.1f}%   ({report.n_trades} trades)")

        if report.group_contributions:
            print(f"\n  ── Layer 1: Feature Group Attribution ──────────────")
            print(f"  (positive = this signal family drove profitable trades)")
            for grp, contrib in report.group_contributions.items():
                bar_len = int(abs(contrib) * 400)
                bar     = ('█' * bar_len)[:20]
                sign    = '+' if contrib >= 0 else '-'
                print(f"  {grp:<14} {sign}{abs(contrib):.4f}  {bar}")
            if report.top_features:
                print(f"\n  Top 5 individual features:")
                for feat, contrib in report.top_features[:5]:
                    grp = FEATURE_GROUPS.get(feat, 'Other')
                    bar = ('█' * int(abs(contrib) * 400))[:15]
                    print(f"    {feat:<20} {contrib:>+.4f}  {bar}  [{grp}]")

        if report.sector_attribution:
            print(f"\n  ── Layer 2: Sector Attribution ─────────────────────")
            print(f"  {'Sector':<22} {'Trades':>6} {'WinRate':>8} {'AvgPnL':>8} {'SelectEff':>10}")
            for sec, d in report.sector_attribution.items():
                print(f"  {sec:<22} {d['n_trades']:>6} {d['win_rate']:>7.1f}% "
                      f"{d['avg_pnl_pct']:>+7.2f}% {d['selection_effect']:>+9.3f}")

        if report.confidence_buckets:
            print(f"\n  ── Layer 3: Confidence Attribution ─────────────────")
            print(f"  {'Bucket':<10} {'Trades':>6} {'WinRate':>8} {'AvgPnL':>8}")
            for lbl, d in report.confidence_buckets.items():
                marker = ' ← best' if d['avg_pnl_pct'] == max(
                    v['avg_pnl_pct'] for v in report.confidence_buckets.values()) else ''
                print(f"  {lbl:<10} {d['n_trades']:>6} {d['win_rate']:>7.1f}% "
                      f"{d['avg_pnl_pct']:>+7.2f}%{marker}")

        if report.monthly_pnl:
            print(f"\n  ── Layer 4: Timing Attribution ─────────────────────")
            for month, d in report.monthly_pnl.items():
                bar_len = min(int(abs(d['total_pnl_pct']) * 2), 20)
                bar     = ('█' * bar_len) if d['total_pnl_pct'] >= 0 else ('░' * bar_len)
                print(f"  {month}  {d['total_pnl_pct']:>+7.2f}%  {bar}")
            print(f"  Best month: {report.best_month}  |  Worst: {report.worst_month}")

        if report.rule_based_attribution:
            print(f"\n  ── Layer 5: Rule-Based Strategy Attribution ────────")
            ml_avg = report.rule_based_attribution.get('ML_only', {}).get('avg_pnl', 0)
            rb_avg = report.rule_based_attribution.get('With_RuleBase', {}).get('avg_pnl', 0)
            delta  = rb_avg - ml_avg
            print(f"  ML-only trades avg PnL     : {ml_avg:>+7.2f}%")
            print(f"  Rule-base co-voted avg PnL : {rb_avg:>+7.2f}%")
            print(f"  Delta (rule-base effect)   : {delta:>+7.2f}% "
                  f"{'✅ additive' if delta > 0 else '❌ detractor'}")

        if report.model_attribution:
            print(f"\n  ── Layer 6: Model Attribution ──────────────────────")
            print(f"  {'Model':<20} {'Trades':>6} {'ConfCorr':>9} {'HighConfPnL':>12}")
            for mdl, d in report.model_attribution.items():
                print(f"  {mdl:<20} {d['n_trades']:>6} {d['pnl_corr']:>+8.3f} "
                      f"{d['avg_pnl_when_high_conf']:>+11.2f}%")

        print(f"{'='*62}\n")