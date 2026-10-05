import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from app.errors import ModelUnavailableError

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "artifacts/thermal.joblib"

class WarmthModel:
    def __init__(self, path=DEFAULT_MODEL):
        self.path = Path(path)
        self.pipeline = None
        self.version = None
        self.min_temperature = None
        self.max_temperature = None
        try:
            meta = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
            if meta["sklearn_version"] != sklearn.__version__:
                raise ValueError("Версия библиотеки отличается")
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError("Артефакт поврежден")
            # Только собственный файл из app.train. Хэш не доказывает доверие.
            pipeline = joblib.load(self.path)
            if not isinstance(pipeline, Pipeline) or set(pipeline.classes_) != {"light", "medium", "warm"}:
                raise ValueError("Неверные классы или тип модели")
            if not isinstance(meta["model_version"], str) or not meta["model_version"]:
                raise ValueError("Нет версии")
            low, high = float(meta["min_temperature"]), float(meta["max_temperature"])
            if not (-50 <= low < high <= 50):
                raise ValueError("Неверная область модели")
            self.pipeline = pipeline
            self.version = meta["model_version"]
            self.min_temperature, self.max_temperature = low, high
        except Exception as exc:
            raise ModelUnavailableError("MODEL_NOT_READY") from exc

    def predict(self, temperature: float) -> dict[str, float]:
        try:
            probabilities = self.pipeline.predict_proba([[temperature]])[0]
            if (len(probabilities) != 3 or not np.isfinite(probabilities).all()
                    or (probabilities < 0).any() or (probabilities > 1).any()
                    or not np.isclose(probabilities.sum(), 1)):
                raise ValueError("Некорректные оценки модели")
            return dict(zip(map(str, self.pipeline.classes_), map(float, probabilities)))
        except Exception as exc:
            raise ModelUnavailableError("MODEL_INFERENCE_FAILED") from exc
