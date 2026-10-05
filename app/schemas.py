from typing import Annotated, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

Category = Literal["top", "bottom", "shoes", "outerwear"]
Occasion = Literal["casual", "work", "sport"]
Precipitation = Literal["none", "rain", "snow"]
Warmth = Literal["light", "medium", "warm"]
ItemId = Annotated[str, Field(pattern=r"^ITEM_[0-9]{1,8}$")]
UserId = Annotated[str, Field(pattern=r"^USER_[0-9]{1,8}$")]

class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

class WeatherContext(Schema):
    temperature: float = Field(ge=-50, le=50)
    precipitation: Precipitation = "none"

class RecommendRequest(Schema):
    user_id: UserId
    weather: WeatherContext
    occasion: Occasion = "casual"
    exclude_item_ids: list[ItemId] | None = Field(default=None, max_length=100)

class ClothingItem(Schema):
    item_id: ItemId
    user_id: UserId
    name: str = Field(min_length=1, max_length=100)
    category: Category
    warmth: Warmth
    occasions: list[Occasion] = Field(min_length=1)
    precipitation: list[Precipitation] = Field(min_length=1)
    min_temperature: float = Field(ge=-50, le=50)
    max_temperature: float = Field(ge=-50, le=50)

    @model_validator(mode="after")
    def check_range(self):
        if self.min_temperature > self.max_temperature:
            raise ValueError("Нижняя граница температуры выше верхней")
        return self

class OutfitItem(Schema):
    item_id: ItemId
    category: Category
    score: float = Field(ge=0, le=1)

class RecommendResponse(Schema):
    request_id: UUID
    user_id: UserId
    recommendation: Literal["OUTFIT_READY", "NEEDS_USER_FEEDBACK"]
    confidence: float | None = Field(ge=0, le=1)
    outfit: list[OutfitItem]
    model_version: str = Field(min_length=1)
    needs_user_feedback: bool
    reason: str

    @model_validator(mode="after")
    def check_status(self):
        categories = [item.category for item in self.outfit]
        if len(categories) != len(set(categories)):
            raise ValueError("Категории в комплекте не должны повторяться")
        if self.recommendation == "OUTFIT_READY":
            if (self.needs_user_feedback or self.confidence is None
                    or self.confidence < 0.5
                    or not {"top", "bottom", "shoes"}.issubset(categories)):
                raise ValueError("Некорректный готовый комплект")
        elif not self.needs_user_feedback:
            raise ValueError("Отказ требует участия пользователя")
        return self
