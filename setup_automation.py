"""
setup_automation.py
Automatically creates Windows Task Scheduler tasks for QuantAI.
Run ONCE as Administrator:  python setup_automation.py

Tasks created:
  QuantAI\\DataPipeline   — weekdays 8:00 AM   — fetch latest prices
  QuantAI\\PaperTrade     — weekdays 3:45 PM   — scan + ML signals after market close
  QuantAI\\WeeklyRetrain  — Sunday   9:00 AM   — retrain all models with fresh data
"""
import subprocess, os, sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    {
        "name"  : "QuantAI\\DataPipeline",
        "bat"   : os.path.join(PROJECT_DIR, "run_pipeline.bat"),
        "time"  : "08:00",
        "days"  : "MON,TUE,WED,THU,FRI",
        "desc"  : "Fetches latest NSE price data from Yahoo Finance"
    },
    {
        "name"  : "QuantAI\\PaperTrade",
        "bat"   : os.path.join(PROJECT_DIR, "run_paper_trade.bat"),
        "time"  : "15:45",
        "days"  : "MON,TUE,WED,THU,FRI",
        "desc"  : "Runs ensemble ML scan after NSE market close (3:30 PM IST)"
    },
    {
        "name"  : "QuantAI\\WeeklyRetrain",
        "bat"   : os.path.join(PROJECT_DIR, "run_train_xgboost_all.bat"),
        "time"  : "09:00",
        "days"  : "SUN",
        "desc"  : "Retrains all XGBoost models with the week's fresh data"
    },
]

def create_task(task):
    cmd = [
        "schtasks", "/create", "/f",
        "/tn", task["name"],
        "/tr", f'"{task["bat"]}"',
        "/sc", "WEEKLY",
        "/d", task["days"],
        "/st", task["time"],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print(f"  ✅  {task['name']:<35} → {task['time']} on {task['days']}")
        print(f"       {task['desc']}")
    else:
        print(f"  ❌  {task['name']} — {result.stderr.strip()}")
        print(f"       Try running as Administrator (right-click → Run as Administrator)")

def delete_task(name):
    subprocess.run(["schtasks", "/delete", "/f", "/tn", name],
                   capture_output=True, shell=True)

print("\n" + "="*60)
print("  QuantAI — Windows Task Scheduler Setup")
print("="*60)
print()

for task in TASKS:
    delete_task(task["name"])   # remove old version first
    create_task(task)
    print()

print("="*60)
print("\n  To verify tasks were created:")
print("  → Open Task Scheduler (Start menu → search 'Task Scheduler')")
print("  → Look under Task Scheduler Library → QuantAI")
print()
print("  To run a task immediately:")
print("  → Right-click the task → Run")
print()
print("  To remove all tasks:")
print("  → schtasks /delete /f /tn QuantAI\\DataPipeline")
print("  → schtasks /delete /f /tn QuantAI\\PaperTrade")
print("  → schtasks /delete /f /tn QuantAI\\WeeklyRetrain")
print("="*60 + "\n")
