import math
from uuid import uuid4
from app.errors import ModelUnavailableError
from app.schemas import RecommendRequest, RecommendResponse, OutfitItem

class Recommender:
    def __init__(self, model, wardrobe):
        self.model = model
        self.wardrobe = wardrobe

    def recommend(self, request: RecommendRequest | dict) -> RecommendResponse:
        data = request.model_dump() if isinstance(request, RecommendRequest) else request
        request = RecommendRequest.model_validate(data)
        temperature = request.weather.temperature
        result = dict(request_id=uuid4(), user_id=request.user_id,
                      model_version=self.model.version)
        if not self.model.min_temperature <= temperature <= self.model.max_temperature:
            return RecommendResponse(**result, recommendation="NEEDS_USER_FEEDBACK",
                confidence=None, outfit=[], needs_user_feedback=True,
                reason="Температура вне диапазона учебной модели")
        try:
            scores = self.model.predict(temperature)
            if (set(scores) != {"light", "medium", "warm"}
                    or any(isinstance(x, bool) or not isinstance(x, (float, int))
                           or not math.isfinite(x) or not 0 <= x <= 1 for x in scores.values())
                    or not math.isclose(sum(scores.values()), 1, abs_tol=1e-6)):
                raise ValueError("Некорректные оценки")
        except Exception as exc:
            raise ModelUnavailableError("MODEL_INFERENCE_FAILED") from exc
        items = self.wardrobe.for_user(request.user_id)
        excluded = set(request.exclude_item_ids or [])
        candidates = [item for item in items
            if item.item_id not in excluded
            and request.occasion in item.occasions
            and request.weather.precipitation in item.precipitation
            and item.min_temperature <= temperature <= item.max_temperature]
        required = ["top", "bottom", "shoes"]
        if temperature <= 15 or request.weather.precipitation != "none":
            required.append("outerwear")
        outfit = []
        for category in required:
            matches = [item for item in candidates if item.category == category]
            if not matches:
                return RecommendResponse(**result, recommendation="NEEDS_USER_FEEDBACK",
                    confidence=None, outfit=[], needs_user_feedback=True,
                    reason=f"Нет подходящей вещи категории {category}")
            # При равных оценках результат воспроизводим: выбираем меньший ID.
            best = min(matches, key=lambda item: (-scores[item.warmth], item.item_id))
            outfit.append(OutfitItem(item_id=best.item_id, category=best.category,
                                     score=scores[best.warmth]))
        confidence = min(item.score for item in outfit)
        manual = confidence < 0.5
        return RecommendResponse(**result,
            recommendation="NEEDS_USER_FEEDBACK" if manual else "OUTFIT_READY",
            confidence=confidence, outfit=outfit, needs_user_feedback=manual,
            reason="Подтвердите комплект: оценка ниже 0.5" if manual else "Комплект подобран")
