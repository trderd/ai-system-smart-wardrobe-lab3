import json
from pathlib import Path
from app.errors import WardrobeUnavailableError
from app.schemas import ClothingItem

DEFAULT_WARDROBE = Path(__file__).resolve().parents[1] / "data/wardrobe.json"

class WardrobeRepository:
    def __init__(self, path=DEFAULT_WARDROBE):
        self.path = Path(path)

    def for_user(self, user_id: str) -> list[ClothingItem]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                raise ValueError("Ожидался список вещей")
            items = [ClothingItem.model_validate(row) for row in raw]
            keys = [(item.user_id, item.item_id) for item in items]
            if len(keys) != len(set(keys)):
                raise ValueError("Повтор идентификатора вещи")
            return [item for item in items if item.user_id == user_id]
        except Exception as exc:
            raise WardrobeUnavailableError("WARDROBE_NOT_READY") from exc
