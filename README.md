# QuantAI — Quant Trading Dashboard for NSE

QuantAI is a self-hosted quantitative trading platform for NSE (India) equities. It combines an ensemble of ML models (Random Forest, XGBoost, sequence models, Transformer, RL agent) with rule-based signals, risk analytics, an options desk, and a paper-trading simulator — all served through a FastAPI backend and a lightweight HTML/JS dashboard.

> **Not financial advice.** QuantAI is a research and paper-trading tool. Nothing it outputs is a recommendation to buy or sell real securities.

---

## Features

| Page | What it does |
|---|---|
| **Signals** | Live BUY/SELL/HOLD signals for 50 NSE stocks, ranked by ensemble confidence |
| **Stock detail** | Price chart, model breakdown (RF/XGBoost/SeqNN/Transformer/RL), news & sentiment, regime |
| **Portfolio / Risk** | Tail Risk Index, correlation matrix, drawdown recovery, risk parity |
| **Options Desk** | Live NSE chain where available, theoretical Black-Scholes chain as fallback, strategy builder, AI strategy suggestion, opportunity scanner |
| **Paper Trade** | Simulated trading with ₹1,00,000 starting capital — no real money, no broker connection |
| **Performance** | Win rate, P&L history, per-trade breakdown |
| **System** | API health, scheduler status, cache status |

---

## Tech stack

- **Backend:** FastAPI + Uvicorn, SQLite, scikit-learn, XGBoost, yfinance
- **Frontend:** Static HTML/CSS/JS (no build step, no framework) — open directly in a browser or serve as static files
- **Models:** Random Forest, XGBoost, a sequence model (SeqNN/LSTM), a Transformer, and a Q-learning RL agent, combined into a weighted ensemble

---

## Prerequisites

- Python 3.10 – 3.13 (avoid 3.14 — some dependencies don't have Windows wheels for it yet)
- Internet access (yfinance pulls live/historical NSE price data)
- Git (only needed if you're pushing to GitHub / deploying)

---

## Quick start (local)

```bash
# 1. Clone and set up
git clone <your-repo-url>
cd QuantAI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 2. One-time setup — builds the price DB, trains core models, seeds sample data
python pipeline.py
python train_all_models.py
python train_xgboost_all.py
python seed_trades.py
python create_admin.py       # creates your login — you'll be prompted for username/email/password
```

**Optional extra models** (not required — the dashboard works fine without them, it just shows "not trained" for whichever you skip):
```bash
python train_lstm_all.py          # SeqNN, ~10 min
python train_transformer_all.py   # Transformer, ~10 min
python train_rl_agent.py          # RL Agent, ~15 min
python fetch_fii_dii.py           # FII/DII institutional flow data
```

**Every time you want to use it:**
```bash
python -m uvicorn src.api:app --reload --port 8000
```
Then just open **http://127.0.0.1:8000** in your browser — the dashboard loads directly from the same server, no separate file to open.

Sanity checks if something looks off: `http://127.0.0.1:8000/api/status`, `/signals`, `/tail-risk`.

---

## Project structure

```
QuantAI/
├── src/                 # Backend: API, models, ensemble, risk, options, auth
├── dashboard/           # Frontend: static HTML/CSS/JS pages
│   └── assets/          # api.js (API client), shell.js, theme.css, charts
├── data/                # SQLite DB + cached data (gitignored — generated locally)
├── models/              # Trained .pkl model files (gitignored — generated locally)
├── pipeline.py           # Builds the price database
├── train_*.py             # Model training scripts
├── create_admin.py        # Creates your first login
├── seed_trades.py         # Seeds sample trade history for the Performance page
├── requirements.txt
├── render.yaml             # Render Blueprint config (see Deployment below)
└── Procfile                 # Alternative start command for Render/Heroku-style hosts
```

---

## Deploying to Render

The backend now serves the dashboard itself — visiting your Render URL opens the dashboard directly, no separate static site needed, and no manual editing of `api.js` required. One service, one link.

### Step 1 — Push to GitHub

If you haven't already, get the project into a GitHub repo (VS Code's Source Control panel → Publish to GitHub is the easiest way). Make sure your `.gitignore` excludes `venv/`, `data/`, `models/`, and `.env` — you don't want a multi-GB venv or your local database in the repo.

### Step 2 — Create the Render service

This project already ships with a `render.yaml`, which Render can read automatically:

1. Go to [render.com](https://render.com) → sign in → **New** → **Blueprint**.
2. Connect your GitHub account and select your `QuantAI` repo.
3. Render detects `render.yaml` and pre-fills everything:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT --workers 1`
   - A persistent disk mounted at `/app/data` (1 GB) — this is what keeps your SQLite database alive across deploys/restarts, instead of it vanishing every time Render redeploys.
   - An auto-generated `QUANTAI_SECRET` environment variable.
4. Click **Apply** / **Create Blueprint**.

If you'd rather set it up manually instead of via Blueprint: **New → Web Service**, connect the repo, and fill in the same build/start commands yourself, then add a disk under the service's **Disks** tab (mount path `/app/data`, size 1 GB).

> **Pick at least the Starter plan, not Free.** Two reasons: Render's Free tier spins your service down after 15 minutes of inactivity (bad for a dashboard you want available anytime — every cold start takes 30–60+ seconds), and Free tier doesn't include Shell access, which you need for the one-time setup in the next step.

### Step 3 — Run one-time setup on the live server

Your Render disk starts empty — it doesn't have your price database or trained models yet. Run the same setup scripts you used locally, but on the Render server itself:

1. Once your first deploy finishes, go to your service in the Render dashboard → **Shell** tab.
2. Run, in order:
   ```bash
   python pipeline.py
   python train_all_models.py
   python train_xgboost_all.py
   python seed_trades.py
   python create_admin.py
   ```

> **Know this limitation before you rely on it:** the persistent disk in `render.yaml` only covers `data/` (your SQLite database). `models/` — your trained `.pkl` files — is **not** on the persistent disk, because Render only supports one disk mount per service, and `data/` and `models/` are separate top-level folders. In practice this means: your database survives redeploys, but **your trained models get wiped on every redeploy**, and you'll need to re-run the `train_*.py` commands above again each time you push a change and Render redeploys. For occasional updates this is a minor annoyance (a few minutes' wait); if you're deploying frequently, consider restructuring so `models/` lives under `data/models/` instead (would need updating `MODELS_DIR` in `src/ensemble_model.py`, `src/lstm_model.py`, `src/model.py`, `src/tft_model.py`, `src/transformer_model.py`, and `src/xgboost_model.py`) so one disk mount covers both — happy to do that for you if you want it.

### Step 4 — Open your link

That's it — visit your Render URL (e.g. `https://quantai-api.onrender.com`) and the dashboard loads directly. The frontend auto-detects it's being served from the same origin as the API and routes all requests there automatically — nothing to configure.

Want a nicer link? Rename the service (Render dashboard → Settings) to get a custom subdomain like `https://quantai.onrender.com`.

### Step 5 — Verify

`https://<your-url>/api/status` should return a clean JSON response (this moved from `/` so the dashboard could live there instead). Then sign in on the dashboard with the admin account you created in Step 3.

<details>
<summary>Prefer two separate services instead? (optional, not necessary)</summary>

If you'd rather keep the dashboard and API as separate Render services (e.g. to scale or update them independently), you still can:
1. Deploy the API as above.
2. Deploy a second **Static Site** pointed at the `dashboard/` folder.
3. Set `window.QUANTAI_API_BASE = "https://your-api-url.onrender.com"` in a small inline script before `api.js` loads on each dashboard page, since they'd no longer share an origin.

This isn't necessary for most setups — the single-service approach above is simpler and is what `render.yaml` is configured for.
</details>

---

## Setting up Telegram alerts (per-user)

Any user can connect their own Telegram account from **Settings → Telegram Alerts → Add Telegram Bot** — no manual chat_id needed, just click and hit Start in Telegram. Here's the one-time setup to make that work:

1. **Create a bot** — message [@BotFather](https://t.me/BotFather) on Telegram, send `/newbot`, follow the prompts. You'll get a bot token and a bot username (e.g. `@QuantAI_Alerts_Bot`).
2. **Set environment variables**:
   ```
   TELEGRAM_BOT_TOKEN=<the token BotFather gave you>
   TELEGRAM_BOT_USERNAME=<your bot's username, without the @>
   ```
   Add these to your local `.env`, and to Render's **Environment** tab if deployed.
3. **Register the webhook** (only works against a public HTTPS URL — won't work on localhost):
   ```bash
   python setup_telegram_webhook.py https://your-deployed-url.onrender.com
   ```
   Run this once, after your app is deployed and those environment variables are set.
4. Done — any user can now click **Add Telegram Bot** in Settings, hit Start in the Telegram chat that opens, and they'll start receiving alerts.

> If you're migrating from an older version of this project, note that `alerts.py` used to have a real bot token and a single hardcoded `CHAT_ID` directly in the source file. If that file was ever pushed to a public repo, **revoke that token via @BotFather immediately** and generate a fresh one — the token in this version comes only from `TELEGRAM_BOT_TOKEN`, never committed to source control.

---

## Environment variables

Copy `.env.example` to `.env` for local use, or set these directly in Render's **Environment** tab:

| Variable | Purpose | Required? |
|---|---|---|
| `QUANTAI_SECRET` | Session/token secret | Auto-generated by Render, not currently required for anything else |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot's token (from @BotFather) | Needed for Telegram alerts |
| `TELEGRAM_BOT_USERNAME` | Your bot's @username, without the @ | Needed for the "Add Telegram Bot" connect link |

---

## Troubleshooting

- **"Can't reach the QuantAI API"** → the backend isn't running. If you're opening a dashboard HTML file directly by double-click (`file://`) instead of through the server, it falls back to expecting the API at `http://127.0.0.1:8000` — start it locally, or set `window.QUANTAI_API_BASE` to your deployed URL before `api.js` loads.
- **Model breakdown shows "not trained"** → you haven't run that model's training script yet (see Quick Start above); this is cosmetic, not an error.
- **Options chain shows a "🧮 Theoretical" badge** → expected. NSE options aren't available through our data provider (yfinance), so the chain is modeled via Black-Scholes off real spot price + real historical volatility instead. Open Interest/Volume/PCR/Max Pain are intentionally left blank rather than faked.
- Check the actual terminal/Shell output for tracebacks when something 500s — that's almost always more informative than the browser's error message.

---

## License

Add your preferred license here (MIT, etc.) before making the repo public, if you haven't already.
