"""
src/drawdown_recovery.py

Drawdown Recovery Manager — Auto-Reduce on Loss.

The gap in the existing risk system:
  src/risk.py already halts trading at -15% portfolio drawdown (the
  hard floor). But between 0% and -15% there is nothing — positions
  are full-sized even after a string of losses, even when the portfolio
  has already dipped -10%. That's the problem this module solves.

What this does:
  Tracks TWO independent stress signals simultaneously:

    Signal A — Portfolio drawdown depth (from peak capital)
      How far below the running high-water mark are we right now?
      Even a single large loss can trigger this.

    Signal B — Consecutive loss streak
      How many trades IN A ROW have been losses, regardless of size?
      A string of small losses erodes confidence and capital equally.

  Both signals independently produce a size multiplier (0.0 – 1.0).
  The LOWER of the two wins — so either a deep drawdown OR a bad
  streak is enough to trigger size reduction. Both must recover before
  full size returns.

  Drawdown tiers (Signal A):
    0%  to  -3%  → 1.00  (full size — business as usual)
    -3% to  -5%  → 0.80  (early warning — slight reduction)
    -5% to  -8%  → 0.60  (moderate — meaningful reduction)
    -8% to -11%  → 0.40  (significant — survival sizing)
    -11% to -14% → 0.20  (critical — minimum size only)
    -14%+        → 0.00  (halt — before the -15% hard stop in risk.py)

  Consecutive loss tiers (Signal B):
    0 – 1 losses → 1.00  (normal)
    2 losses     → 0.85
    3 losses     → 0.65
    4 losses     → 0.45
    5+ losses    → 0.25  (survival mode — something systematic is wrong)

  Recovery path (how you get back to full size):
    A multiplier below 1.0 does NOT snap back after one win. It ratchets
    up gradually — each consecutive win after the loss streak adds one
    "recovery step." This prevents yo-yo sizing in choppy markets.
    You need at least 2 consecutive wins to begin recovery, and the
    step size is +15% per consecutive win, capped at 1.0.
    A single loss resets the win streak and pauses recovery.

State persistence:
  All state is saved to data/recovery_state.json after every session
  so drawdown tracking survives restarts. Load it at the start of each
  paper_trade.py session via DrawdownRecoveryManager.load_state().
"""

import json
import os
import numpy as np# type: ignore
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Tier definitions ──────────────────────────────────────

DRAWDOWN_TIERS = [
    # (threshold_pct, size_multiplier, tier_label)
    (-3.0,  1.00, 'NORMAL'),
    (-5.0,  0.80, 'CAUTION'),
    (-8.0,  0.60, 'REDUCED'),
    (-11.0, 0.40, 'DEFENSIVE'),
    (-14.0, 0.20, 'CRITICAL'),
    (-99.0, 0.00, 'HALTED'),
]

CONSEC_LOSS_TIERS = [
    # (max_consecutive_losses, size_multiplier, tier_label)
    (1, 1.00, 'NORMAL'),
    (2, 0.85, 'WATCH'),
    (3, 0.65, 'STREAK'),
    (4, 0.45, 'BAD_STREAK'),
    (99, 0.25, 'CRISIS'),
]

RECOVERY_STEP       = 0.15   # multiplier gain per consecutive win
MIN_WINS_TO_RECOVER = 2      # need at least this many consecutive wins to start recovery
STATE_PATH          = os.path.join('data', 'recovery_state.json')


@dataclass
class RecoveryState:
    """
    Full state of the recovery manager — saved to disk after every session.
    """
    peak_capital        : float  = 100_000.0
    current_capital     : float  = 100_000.0
    current_drawdown_pct: float  = 0.0

    consecutive_losses  : int    = 0
    consecutive_wins    : int    = 0
    total_losses        : int    = 0
    total_wins          : int    = 0

    dd_multiplier       : float  = 1.0   # from drawdown tiers
    streak_multiplier   : float  = 1.0   # from consecutive loss tiers
    size_multiplier     : float  = 1.0   # min(dd_mult, streak_mult) — the one used
    recovery_bonus      : float  = 0.0   # added back during recovery

    dd_tier             : str    = 'NORMAL'
    streak_tier         : str    = 'NORMAL'
    in_recovery         : bool   = False

    last_updated        : str    = ''
    history             : list   = field(default_factory=list)  # last 30 trade outcomes


class DrawdownRecoveryManager:
    """
    Tracks loss streaks and drawdown depth, and converts them into a
    single size_multiplier (0.0–1.0) that paper_trade.py multiplies
    against the shares from RiskManager.calculate_position_size().

    Usage in paper_trade.py:
        drm = DrawdownRecoveryManager()
        drm.load_state()

        # After closing a trade:
        drm.update(result='WIN', pnl_pct=4.2, current_capital=103500, peak_capital=104000)

        # Before sizing a new trade:
        shares = int(base_shares * drm.state.size_multiplier)
    """

    def __init__(self, initial_capital: float = 100_000.0):
        self.state = RecoveryState(
            peak_capital    = initial_capital,
            current_capital = initial_capital,
        )

    # ── Tier lookups ──────────────────────────────────────

    def _dd_tier(self, drawdown_pct: float) -> tuple[float, str]:
        """Returns (multiplier, label) for a given drawdown percentage."""
        for threshold, mult, label in DRAWDOWN_TIERS:
            if drawdown_pct >= threshold:
                return mult, label
        return 0.0, 'HALTED'

    def _streak_tier(self, consecutive_losses: int) -> tuple[float, str]:
        """Returns (multiplier, label) for a given consecutive loss count."""
        for max_losses, mult, label in CONSEC_LOSS_TIERS:
            if consecutive_losses <= max_losses:
                return mult, label
        return 0.25, 'CRISIS'

    # ── Core update ───────────────────────────────────────

    def update(self,
               result        : str,    # 'WIN' or 'LOSS'
               pnl_pct       : float,  # % gain or loss on this trade
               current_capital: float,
               peak_capital  : float) -> float:
        """
        Called AFTER a trade closes (stop loss hit or manual exit).
        Updates the internal state and returns the NEW size_multiplier
        that applies to all subsequent trades this session.

        Parameters
        ----------
        result          : 'WIN' or 'LOSS'
        pnl_pct         : realised PnL as a percentage (e.g. +3.2 or -2.1)
        current_capital : portfolio value right now
        peak_capital    : highest portfolio value ever reached

        Returns
        -------
        float : the new size_multiplier (0.0 – 1.0)
        """
        s = self.state

        # ── Update capital tracking ───────────────────────
        s.current_capital  = current_capital
        s.peak_capital     = max(peak_capital, current_capital)
        s.current_drawdown_pct = (
            (current_capital - s.peak_capital) / s.peak_capital * 100
        )

        # ── Update streak counters ────────────────────────
        if result == 'WIN':
            s.consecutive_wins  += 1
            s.consecutive_losses = 0
            s.total_wins        += 1
        else:
            s.consecutive_losses += 1
            s.consecutive_wins   = 0
            s.total_losses      += 1

        # ── Compute tier multipliers ──────────────────────
        s.dd_multiplier,     s.dd_tier     = self._dd_tier(s.current_drawdown_pct)
        s.streak_multiplier, s.streak_tier = self._streak_tier(s.consecutive_losses)

        # ── Apply recovery bonus (only if both tiers are below 1.0) ──
        raw_mult = min(s.dd_multiplier, s.streak_multiplier)

        if raw_mult < 1.0:
            s.in_recovery = True
            # Recovery: consecutive wins (after the minimum threshold)
            # each add RECOVERY_STEP, but can't exceed the tier ceiling
            wins_qualifying = max(0, s.consecutive_wins - MIN_WINS_TO_RECOVER + 1)
            s.recovery_bonus = min(
                wins_qualifying * RECOVERY_STEP,
                1.0 - raw_mult          # can't recover past 1.0
            )
        else:
            s.in_recovery    = False
            s.recovery_bonus = 0.0

        s.size_multiplier = min(1.0, raw_mult + s.recovery_bonus)

        # ── Append to rolling history (keep last 30) ──────
        s.history.append({
            'date'          : str(date.today()),
            'result'        : result,
            'pnl_pct'       : round(pnl_pct, 2),
            'drawdown_pct'  : round(s.current_drawdown_pct, 2),
            'consec_losses' : s.consecutive_losses,
            'consec_wins'   : s.consecutive_wins,
            'size_multiplier': round(s.size_multiplier, 3),
            'dd_tier'       : s.dd_tier,
            'streak_tier'   : s.streak_tier,
        })
        if len(s.history) > 30:
            s.history = s.history[-30:]

        s.last_updated = datetime.now().isoformat()
        return s.size_multiplier

    def update_capital(self, current_capital: float, peak_capital: float):
        """
        Called at the START of each session (before any trades) to
        refresh the drawdown multiplier based on current portfolio value,
        even if no trade closed today. This ensures the multiplier stays
        accurate on days where open positions are losing on paper but
        haven't been closed yet.
        """
        s = self.state
        s.current_capital  = current_capital
        s.peak_capital     = max(peak_capital, current_capital)
        s.current_drawdown_pct = (
            (current_capital - s.peak_capital) / s.peak_capital * 100
        )

        # Recompute dd multiplier; keep streak multiplier as-is
        s.dd_multiplier, s.dd_tier = self._dd_tier(s.current_drawdown_pct)
        raw_mult = min(s.dd_multiplier, s.streak_multiplier)

        wins_qualifying  = max(0, s.consecutive_wins - MIN_WINS_TO_RECOVER + 1)
        s.recovery_bonus = min(wins_qualifying * RECOVERY_STEP, 1.0 - raw_mult) \
                           if raw_mult < 1.0 else 0.0

        s.size_multiplier = min(1.0, raw_mult + s.recovery_bonus)

    # ── State persistence ─────────────────────────────────

    def save_state(self, path: str = STATE_PATH):
        os.makedirs('data', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self.state), f, indent=2)

    def load_state(self, path: str = STATE_PATH) -> bool:
        """
        Loads persisted state. Returns True if file existed, False if
        starting fresh (first run or file deleted).
        """
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            s = self.state
            for k, v in data.items():
                if hasattr(s, k):
                    setattr(s, k, v)
            return True
        except Exception:
            return False

    # ── Reporting ─────────────────────────────────────────

    def get_report(self) -> dict:
        """Returns a clean dict snapshot for display / API / alerts."""
        s = self.state
        win_rate = (s.total_wins / (s.total_wins + s.total_losses) * 100
                    if (s.total_wins + s.total_losses) > 0 else 0.0)
        return {
            'size_multiplier'    : round(s.size_multiplier, 3),
            'dd_tier'            : s.dd_tier,
            'streak_tier'        : s.streak_tier,
            'current_drawdown_pct': round(s.current_drawdown_pct, 2),
            'consecutive_losses' : s.consecutive_losses,
            'consecutive_wins'   : s.consecutive_wins,
            'in_recovery'        : s.in_recovery,
            'recovery_bonus'     : round(s.recovery_bonus, 3),
            'dd_multiplier'      : round(s.dd_multiplier, 3),
            'streak_multiplier'  : round(s.streak_multiplier, 3),
            'total_wins'         : s.total_wins,
            'total_losses'       : s.total_losses,
            'win_rate'           : round(win_rate, 1),
            'peak_capital'       : round(s.peak_capital, 2),
            'current_capital'    : round(s.current_capital, 2),
        }

    def print_report(self):
        """Human-readable recovery dashboard — called from paper_trade.py."""
        r   = self.get_report()
        s   = self.state

        mult_pct  = r['size_multiplier'] * 100
        dd_emoji  = {'NORMAL': '🟢', 'CAUTION': '🟡', 'REDUCED': '🟠',
                     'DEFENSIVE': '🔴', 'CRITICAL': '🔴', 'HALTED': '🚫'
                     }.get(r['dd_tier'], '⚪')
        str_emoji = {'NORMAL': '🟢', 'WATCH': '🟡', 'STREAK': '🟠',
                     'BAD_STREAK': '🔴', 'CRISIS': '🔴'
                     }.get(r['streak_tier'], '⚪')

        print(f"\n  {'─'*54}")
        print(f"  ⚖️  DRAWDOWN RECOVERY STATUS")
        print(f"  {'─'*54}")
        print(f"  Size Multiplier    : {mult_pct:>5.1f}%  ←  all positions scaled by this")
        print(f"  Drawdown Tier      : {dd_emoji} {r['dd_tier']:<12} "
              f"({r['current_drawdown_pct']:+.1f}% from peak)")
        print(f"  Streak Tier        : {str_emoji} {r['streak_tier']:<12} "
              f"({r['consecutive_losses']} consecutive loss(es))")
        if r['in_recovery']:
            print(f"  Recovery Mode      : 🔄 YES — {r['consecutive_wins']} win(s) "
                  f"after streak, +{r['recovery_bonus']*100:.0f}% recovered")
        print(f"  Session Record     : "
              f"{r['total_wins']}W / {r['total_losses']}L  ({r['win_rate']:.1f}% WR)")
        if mult_pct < 100:
            needed = (
                f"Reduce drawdown to "
                f"{DRAWDOWN_TIERS[max(0, [t[2] for t in DRAWDOWN_TIERS].index(r['dd_tier'])-1)][0]:.0f}% "
                f"AND win {MIN_WINS_TO_RECOVER}+ in a row to begin recovery"
                if r['dd_tier'] != 'NORMAL' else
                f"Win {MIN_WINS_TO_RECOVER}+ consecutive trades to begin recovery"
            )
            print(f"  To recover         : {needed}")
        print(f"  {'─'*54}\n")