import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    for name in ["summer", "winter", "work_rain", "excluded", "no_outfit", "invalid", "out_of_domain"]:
        print(f"\n=== {name} ===", flush=True)
        completed = subprocess.run([sys.executable, "-m", "app.main", str(ROOT / f"examples/{name}.json")],
                                   text=True, capture_output=True, cwd=ROOT)
        print(completed.stdout or completed.stderr, end="")
        print("exit_code:", completed.returncode)
    print("\n=== missing_model ===", flush=True)
    completed = subprocess.run([sys.executable, "-m", "app.main", str(ROOT / "examples/summer.json"),
                                "--model", str(ROOT / "artifacts/missing.joblib")],
                               text=True, capture_output=True, cwd=ROOT)
    print(completed.stdout or completed.stderr, end="")
    print("exit_code:", completed.returncode)

if __name__ == "__main__":
    main()
