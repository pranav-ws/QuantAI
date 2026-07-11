"""
setup_telegram_webhook.py — run this ONCE after deploying, to tell Telegram
where to send incoming messages (i.e. when a user hits "Start" after
clicking their personal connect link).

This needs a public HTTPS URL — it will NOT work against localhost/
127.0.0.1, since Telegram's servers need to be able to reach it directly.
Run it after your Render deploy (or any other public HTTPS deployment),
pointing at that URL.

Usage:
    python setup_telegram_webhook.py https://your-app.onrender.com

Requires TELEGRAM_BOT_TOKEN to be set in your environment (same one your
app uses — see .env / Render environment variables).
"""
import os
import sys
import requests


def main():
    if len(sys.argv) < 2:
        print("Usage: python setup_telegram_webhook.py https://your-deployed-url.com")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")
    if not base_url.startswith("https://"):
        print("⚠️  Telegram requires HTTPS for webhooks. "
              "A plain http:// URL (or localhost) will not work.")
        sys.exit(1)

    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not bot_token:
        print("❌ TELEGRAM_BOT_TOKEN is not set in your environment. "
              "Set it (same value your deployed app uses) and try again.")
        sys.exit(1)

    webhook_url = f"{base_url}/telegram/webhook"
    resp = requests.post(
        f"https://api.telegram.org/bot{bot_token}/setWebhook",
        json={"url": webhook_url},
        timeout=15,
    )
    data = resp.json()
    if data.get("ok"):
        print(f"✅ Webhook registered: {webhook_url}")
        print("   Users can now click \"Add Telegram Bot\" in the dashboard "
              "and it'll work end-to-end.")
    else:
        print(f"❌ Telegram rejected the webhook setup: {data}")
        sys.exit(1)


if __name__ == "__main__":
    main()
