import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

dashboard_file = PROJECT_ROOT / "dashboard" / "app.py"

with open(dashboard_file, "r", encoding="utf-8") as f:
    code = f.read()

exec(compile(code, str(dashboard_file), "exec"))