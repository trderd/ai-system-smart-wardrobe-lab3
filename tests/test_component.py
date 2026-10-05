import json
import subprocess
import sys
from pathlib import Path
import pytest
from pydantic import ValidationError
from app.errors import ModelUnavailableError, WardrobeUnavailableError
from app.train import train
from app.inference import WarmthModel
from app.wardrobe import WardrobeRepository
from app.schemas import RecommendRequest, RecommendResponse
from app.service import Recommender

ROOT = Path(__file__).resolve().parents[1]

def example(name):
    return json.loads((ROOT / f"examples/{name}.json").read_text())

@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    return train(tmp_path_factory.mktemp("model"))

@pytest.fixture(scope="module")
def component(artifact):
    return Recommender(WarmthModel(artifact), WardrobeRepository())

def test_summer_schema(component):
    result = component.recommend(example("summer"))
    assert isinstance(result, RecommendResponse)
    assert result.recommendation == "OUTFIT_READY"
    assert not result.needs_user_feedback
    assert {x.item_id for x in result.outfit} == {"ITEM_1", "ITEM_2", "ITEM_3"}
    assert result.model_version == "wardrobe-lr-1.0.0"
    assert result.request_id
    assert 0.5 <= result.confidence <= 1

def test_winter_changes_outfit(component):
    summer = component.recommend(example("summer"))
    winter = component.recommend(example("winter"))
    assert winter.recommendation == "OUTFIT_READY"
    assert {x.item_id for x in winter.outfit} == {"ITEM_9", "ITEM_10", "ITEM_11", "ITEM_12"}
    assert winter.outfit != summer.outfit

def test_work_and_rain_rules(component):
    result = component.recommend(example("work_rain"))
    assert result.recommendation == "OUTFIT_READY"
    assert {x.item_id for x in result.outfit} == {"ITEM_20", "ITEM_21", "ITEM_22", "ITEM_23"}

def test_exclusion_prevents_incomplete_success(component):
    result = component.recommend(example("excluded"))
    assert result.needs_user_feedback
    assert result.recommendation == "NEEDS_USER_FEEDBACK"
    assert result.outfit == []
    assert result.confidence is None
    assert "shoes" in result.reason

def test_precipitation_changes_available_items(component):
    request = example("summer")
    request["weather"]["precipitation"] = "rain"
    result = component.recommend(request)
    assert result.needs_user_feedback
    assert result.outfit == []  # Летние кроссовки не разрешены для дождя.

def test_no_sport_outfit(component):
    result = component.recommend(example("no_outfit"))
    assert result.needs_user_feedback and not result.outfit

def test_unknown_user_does_not_get_other_users_clothes(component):
    request = example("summer") | {"user_id": "USER_9999"}
    result = component.recommend(request)
    assert result.needs_user_feedback and not result.outfit

def test_user_ownership(component):
    assert "ITEM_999" not in {x.item_id for x in component.recommend(example("summer")).outfit}

@pytest.mark.parametrize("temperature", [-50, 50])
def test_valid_bounds_controlled_refusal(component, temperature):
    request = example("summer")
    request["weather"]["temperature"] = temperature
    result = component.recommend(request)
    assert result.needs_user_feedback
    assert result.confidence is None
    assert result.outfit == []

@pytest.mark.parametrize("temperature", [-50.1, 50.1, float("nan"), float("inf"), "20", True])
def test_invalid_temperature(component, temperature):
    request = example("summer")
    request["weather"]["temperature"] = temperature
    with pytest.raises(ValidationError):
        component.recommend(request)

@pytest.mark.parametrize("patch", [
    {"user_id": ""}, {"occasion": "unknown"},
    {"exclude_item_ids": ["unknown"]}, {"extra": True},
    {"weather": {"temperature": 20, "precipitation": "storm"}},
])
def test_invalid_contract(component, patch):
    with pytest.raises(ValidationError):
        component.recommend(example("summer") | patch)

def test_missing_artifact(tmp_path):
    with pytest.raises(ModelUnavailableError, match="MODEL_NOT_READY"):
        WarmthModel(tmp_path / "missing.joblib")

def test_corrupt_artifact(artifact, tmp_path):
    path = tmp_path / "thermal.joblib"
    path.write_bytes(artifact.read_bytes() + b"broken")
    path.with_suffix(".json").write_bytes(artifact.with_suffix(".json").read_bytes())
    with pytest.raises(ModelUnavailableError):
        WarmthModel(path)

def test_dependency_failure(component):
    class Broken:
        version = "test"
        min_temperature, max_temperature = -20, 35
        def predict(self, temperature):
            raise RuntimeError("Model failed")
    with pytest.raises(ModelUnavailableError, match="MODEL_INFERENCE_FAILED"):
        Recommender(Broken(), component.wardrobe).recommend(example("summer"))

def test_low_confidence_and_threshold(component):
    class Uncertain:
        version = "test"
        min_temperature, max_temperature = -20, 35
        def predict(self, temperature):
            return {"light": 0.4, "medium": 0.35, "warm": 0.25}
    result = Recommender(Uncertain(), component.wardrobe).recommend(example("summer"))
    assert result.needs_user_feedback and result.confidence == 0.4
    assert len(result.outfit) == 3
    class Boundary(Uncertain):
        def predict(self, temperature):
            return {"light": 0.5, "medium": 0.3, "warm": 0.2}
    result = Recommender(Boundary(), component.wardrobe).recommend(example("summer"))
    assert result.recommendation == "OUTFIT_READY"

@pytest.mark.parametrize("scores", [{"light": float("nan"), "medium": 0.3, "warm": 0.2},
                                     {"light": 1.2, "medium": -0.4, "warm": 0.2},
                                     {"light": 0.2, "medium": 0.2, "warm": 0.2}])
def test_invalid_model_output(component, scores):
    class Invalid:
        version = "test"
        min_temperature, max_temperature = -20, 35
        def predict(self, temperature): return scores
    with pytest.raises(ModelUnavailableError):
        Recommender(Invalid(), component.wardrobe).recommend(example("summer"))

@pytest.mark.parametrize("content", [None, "not JSON", '[{"item_id":"ITEM_1"}]'])
def test_unavailable_wardrobe(component, tmp_path, content):
    path = tmp_path / "wardrobe.json"
    if content is not None: path.write_text(content)
    with pytest.raises(WardrobeUnavailableError):
        Recommender(component.model, WardrobeRepository(path)).recommend(example("summer"))

def test_scaler_is_not_refitted(component):
    scaler = component.model.pipeline.named_steps["scale"]
    before = scaler.mean_.copy()
    first = component.recommend(example("summer"))
    component.recommend(example("winter"))
    second = component.recommend(example("summer"))
    assert (before == scaler.mean_).all()
    assert first.outfit == second.outfit
    assert first.confidence == second.confidence

def test_response_rejects_incomplete_ready():
    from uuid import uuid4
    with pytest.raises(ValidationError):
        RecommendResponse(request_id=uuid4(), user_id="USER_1024", recommendation="OUTFIT_READY",
                          confidence=0.9, outfit=[], model_version="test",
                          needs_user_feedback=False, reason="test")

def test_cli_invalid_request_exit_code():
    result = subprocess.run([sys.executable, "-m", "app.main", "examples/invalid.json"],
                            cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == "VALIDATION_FAILED"
    assert not result.stdout
