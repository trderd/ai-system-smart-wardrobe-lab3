import csv
import hashlib
import json
from pathlib import Path
import joblib
import sklearn
from app.preprocessing import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
VERSION = "wardrobe-lr-1.0.0"
CLASSES = {"light", "medium", "warm"}

def train(output_dir=ROOT / "artifacts"):
    source = ROOT / "data/thermal_train.csv"
    with source.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    temperatures = [float(row["temperature"]) for row in rows]
    labels = [row["warmth"] for row in rows]
    if set(labels) != CLASSES or len(temperatures) != len(set(temperatures)):
        raise ValueError("Нужны три класса и уникальные температуры")
    pipeline = make_pipeline()
    pipeline.fit([[x] for x in temperatures], labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "thermal.joblib"
    joblib.dump(pipeline, path)
    meta = {
        "model_version": VERSION, "sklearn_version": sklearn.__version__,
        "min_temperature": min(temperatures), "max_temperature": max(temperatures),
        "samples": len(rows), "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path

if __name__ == "__main__":
    print(train())
