import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).parent

# Only run scenario scripts (exclude helpers)
EXCLUDE = {
    "run_all.py",
    "calculate_kpis.py",
    "compare_scenarios.py",
}

scripts = sorted(
    p
    for p in THIS_DIR.glob("*.py")
    if p.name not in EXCLUDE and not p.name.startswith("_")
)

if not scripts:
    print("No test scripts found.")
    sys.exit(1)

failed = []

for script in scripts:
    module = f"tests.port_eletrification_studies.{script.stem}"
    print(f"\n▶ Running {module}")

    result = subprocess.run(
        [sys.executable, "-m", module],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if result.returncode != 0:
        failed.append(script.name)

print("\n" + "=" * 50)
if failed:
    print("❌ Failed scripts:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("✅ All tests completed successfully")
