# QuantAI — AI Quant Trading Platform (Windows Edition)

This package contains everything built across Phases 0–7 of the QuantAI
project: data pipeline, feature engineering, ML model, backtester, risk
manager, paper trading engine, FastAPI backend, and a live dashboard —
all adapted to run on **Windows**.

---

## 0. Prerequisites

1. **Python 3.10 – 3.13** (recommended). Download from
   [python.org/downloads](https://www.python.org/downloads/).
   During install, **tick "Add Python to PATH"**.
   > Avoid Python 3.14 for now — some libraries used here don't yet have
   > pre-built Windows wheels for it.
2. **VS Code** (optional but recommended) —
   [code.visualstudio.com](https://code.visualstudio.com/)
3. Internet connection (the scripts download live NSE stock data via
   Yahoo Finance).

---

## 1. Unzip the project

Extract the ZIP anywhere, e.g. `C:\Users\<You>\QuantAI`.
Open that folder in VS Code, or open a terminal (PowerShell / Command
Prompt) and `cd` into it:

```bat
cd C:\Users\<You>\QuantAI
```

Your folder should look like this:

```
QuantAI/
├── data/                  (empty for now — gets filled by the pipeline)
├── models/                (empty for now — gets filled by training)
├── src/
│   ├── __init__.py
│   ├── database.py
│   ├── data_collector.py
│   ├── features.py
│   ├── model.py
│   ├── backtest.py
│   ├── risk.py
│   ├── paper_trader.py
│   ├── api.py
│   └── alerts.py
├── day1.py
├── day1_chart.py
├── explore_features.py
├── pipeline.py
├── query_db.py
├── train_model.py
├── train_all_models.py
├── run_backtest.py
├── test_risk.py
├── paper_trade.py
├── dashboard.html
├── requirements.txt
├── setup_windows.bat
├── run_pipeline.bat
├── run_train_all.bat
├── start_api.bat
├── open_dashboard.bat
├── run_paper_trade.bat
└── README.md  ← you are here
```

---

## 2. One-time setup — install everything

Double-click **`setup_windows.bat`** (or run it from the terminal).

This will:
- Create a virtual environment in `venv\`
- Activate it
- Install every package from `requirements.txt`
  (`yfinance`, `pandas`, `matplotlib`, `mplfinance`, `ta`, `scikit-learn`,
  `joblib`, `fastapi`, `uvicorn`, `python-telegram-bot`, etc.)

This takes a few minutes the first time. Leave the window open until you
see **"Setup complete!"**.

> **Manual alternative**, if you prefer doing it yourself in a terminal:
> ```bat
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
> ```
> Whenever you open a *new* terminal for this project, run
> `venv\Scripts\activate` first — you'll see `(venv)` appear at the start
> of the line when it's active.

---

## 3. Build your local stock database (Phase 1)

Double-click **`run_pipeline.bat`** (or run `python pipeline.py` with the
venv active).

This downloads ~4 years of daily OHLCV data for 10 Nifty 50 stocks
(Reliance, TCS, HDFC Bank, Infosys, ICICI Bank, Hindustan Unilever, SBI,
Bharti Airtel, ITC, Kotak Mahindra Bank) into `data/quantai.db`
(a local SQLite database). Takes 1–2 minutes.

To verify it worked, run:

```bat
python query_db.py
```

You should see a table listing all 10 stocks with ~1,000+ rows each.

---

## 4. Train the ML models (Phase 3)

Double-click **`run_train_all.bat`** (or run `python train_all_models.py`).

This builds 30+ technical-indicator features for every stock, trains a
Random Forest classifier per stock to predict next-day UP/DOWN direction,
and saves each model to `models/<TICKER>_rf_model.pkl`. Takes 1–2 minutes.

You'll see an accuracy score per stock — anything **52%+** is a real
tradeable edge in this context.

> **Optional educational scripts** (run any time with `(venv)` active):
> - `python day1.py` / `python day1_chart.py` — basic data fetch & candlestick chart
> - `python explore_features.py` — visualize indicators for Reliance
> - `python train_model.py` — train + chart a single model (Reliance)
> - `python run_backtest.py` — backtest the Reliance model vs Buy & Hold
> - `python test_risk.py` — demo the risk manager's position sizing

---

## 5. Run a live paper-trading scan (Phase 6)

```bat
python paper_trade.py
```

This scans all 10 stocks **live** (current market data), runs each ML
model, and prints a BUY/SKIP table. Any signal with confidence ≥ 58%
gets sized by the risk manager and logged to:
- `data/paper_capital.json` (current paper capital & start date)
- `data/paper_trades.json` (trade log)

Run this once a day after market close to get tomorrow's signals (see
**Step 8** to automate this).

---

## 6. Start the API server (Phase 7)

Double-click **`start_api.bat`**.

This starts the FastAPI backend at `http://127.0.0.1:8000`. **Keep this
window open** — the dashboard needs it running.

Test it by opening these in your browser:
- `http://127.0.0.1:8000` — status check
- `http://127.0.0.1:8000/signals` — live ML scan of all 10 stocks (JSON)
- `http://127.0.0.1:8000/signal/INFY` — single-stock signal
- `http://127.0.0.1:8000/prices/RELIANCE` — historical OHLCV from the DB
- `http://127.0.0.1:8000/portfolio` — paper trading state

---

## 7. Open the dashboard

With the API server still running, double-click **`open_dashboard.bat`**
(or just double-click `dashboard.html`).

The dashboard shows:
- A live scrolling **ticker tape** of all 10 stocks
- **Stats bar** — buy signal count, stocks scanned, paper capital, total trades
- **Buy Signals panel** — high-confidence (≥58%) picks with RSI/MACD/Volume
- **All Signals table** — every stock ranked by confidence (click a row
  to load its 60-day price chart on the right)
- **Sectors panel** — which sectors currently have active BUY signals
- Auto-refreshes every 5 minutes, or click **REFRESH** any time

If you see a red banner saying it can't reach the API, make sure
`start_api.bat` is still running, then click REFRESH.

---

## 8. Optional — Telegram alerts

Get BUY signals pushed to your phone automatically.

1. In Telegram, message **@BotFather** → send `/newbot` → follow the
   prompts → copy the **bot token** it gives you.
2. Message **@userinfobot** → send `/start` → copy your numeric **Chat ID**.
3. Open `src/alerts.py` and replace:
   ```python
   BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
   CHAT_ID   = "YOUR_CHAT_ID_HERE"
   ```
   with your real values.
4. Open `paper_trade.py` and set:
   ```python
   ENABLE_TELEGRAM_ALERTS = True
   ```
5. Run `python paper_trade.py` — if there's a BUY signal, you'll get a
   Telegram message.

---

## 9. Optional — Back up to GitHub

```bat
git init
git add .
git commit -m "QuantAI v1.0 - Complete quant trading platform"
```

Create a new empty repo at `github.com/new` named `QuantAI`, then:

```bat
git remote add origin https://github.com/YOUR_USERNAME/QuantAI.git
git push -u origin main
```

(`venv/`, the database, models, and trade logs are excluded via
`.gitignore` so the repo stays small — they get regenerated by the
scripts above.)

If `git` isn't recognized, install **Git for Windows** from
[git-scm.com](https://git-scm.com/download/win) first.

---

## 10. Optional — Daily automation (Windows Task Scheduler)

`crontab` doesn't exist on Windows — instead we use **Task Scheduler**
with `run_paper_trade.bat` (already included).

1. Open `run_paper_trade.bat` in a text editor and confirm the folder
   path is correct — `cd /d "%~dp0"` automatically uses the folder the
   `.bat` file lives in, so usually **no edits are needed**.
2. Press the **Start menu**, search for **Task Scheduler**, open it.
3. Click **Create Basic Task** (right panel).
4. Name it `QuantAI Paper Trade` → Next.
5. Trigger: **Weekly** → select Mon, Tue, Wed, Thu, Fri → Next.
6. Time: **4:00 PM** (or whenever you prefer, after market close) → Next.
7. Action: **Start a program** → Browse to your
   `...\QuantAI\run_paper_trade.bat` → Next → Finish.

You can right-click the task → **Run** any time to test it. Output is
appended to `data\log.txt`.

> **PowerShell alternative** (run as Administrator):
> ```powershell
> schtasks /create /tn "QuantAI Paper Trade" /tr "C:\Users\<You>\QuantAI\run_paper_trade.bat" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:00
> ```

---

## Troubleshooting

**`'python' is not recognized...`**
Python isn't on your PATH. Reinstall Python and tick "Add Python to PATH",
or use the **py** launcher (`py -m venv venv`).

**`ModuleNotFoundError: No module named 'yfinance'` (or any package)**
Your virtual environment isn't activated. Run `venv\Scripts\activate`
(you should see `(venv)` at the start of the line), then re-run
`pip install -r requirements.txt`.

**Dashboard shows a red "Can't reach the QuantAI API" banner**
`start_api.bat` isn't running, or was closed. Re-launch it and click
REFRESH on the dashboard.

**`pip install pandas-ta` fails / numba errors**
This project intentionally uses the `ta` library instead of `pandas-ta`
(already set in `requirements.txt`) — no action needed.

**Charts (matplotlib) don't pop up**
Make sure you're running scripts directly (`python day1_chart.py`, etc.)
in a normal terminal — chart windows won't appear if run through some
restricted/remote environments.

**yfinance returns empty data**
NSE markets may be closed (weekends/holidays), or Yahoo Finance is
rate-limiting — wait a few minutes and try again.

---

## What you have

| Component | File(s) |
|---|---|
| Local NSE data pipeline (10 stocks) | `pipeline.py`, `src/database.py`, `src/data_collector.py` |
| 30+ technical indicator feature engine | `src/features.py` |
| Random Forest ML models (per stock) | `src/model.py`, `train_all_models.py` |
| Backtesting engine | `src/backtest.py`, `run_backtest.py` |
| Risk management system | `src/risk.py`, `test_risk.py` |
| Live paper trading scanner | `src/paper_trader.py`, `paper_trade.py` |
| FastAPI REST backend | `src/api.py` |
| Live dashboard | `dashboard.html` |
| Telegram alerts (optional) | `src/alerts.py` |

Enjoy — and remember this is a **paper trading / educational** system,
not financial advice. 🚀
