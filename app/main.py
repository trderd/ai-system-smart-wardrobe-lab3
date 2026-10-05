import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from app.errors import ModelUnavailableError, WardrobeUnavailableError
from app.inference import WarmthModel, DEFAULT_MODEL
from app.wardrobe import WardrobeRepository, DEFAULT_WARDROBE
from app.schemas import RecommendRequest
from app.service import Recommender

def main():
    parser = argparse.ArgumentParser(description="Подбор комплекта одежды")
    parser.add_argument("request", type=Path, help="JSON с запросом")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--wardrobe", type=Path, default=DEFAULT_WARDROBE)
    args = parser.parse_args()
    try:
        data = json.loads(args.request.read_text(encoding="utf-8"))
        request = RecommendRequest.model_validate(data)
        component = Recommender(WarmthModel(args.model), WardrobeRepository(args.wardrobe))
        print(component.recommend(request).model_dump_json(indent=2))
        return 0
    except ValidationError as exc:
        error = {"error": "VALIDATION_FAILED", "detail": exc.errors(include_input=False, include_url=False)}
        code = 2
    except (OSError, UnicodeError, json.JSONDecodeError):
        error, code = {"error": "REQUEST_FILE_INVALID"}, 2
    except (ModelUnavailableError, WardrobeUnavailableError) as exc:
        error, code = {"error": str(exc)}, 3
    print(json.dumps(error, ensure_ascii=False, default=str), file=sys.stderr)
    return code

if __name__ == "__main__":
    raise SystemExit(main())
