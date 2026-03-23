"""Running with `python3 tests/test_grade_chats.py` will run the tests."""

from sae_scoping.evaluation.grade_chats.generic_judges import grade_chats
from pathlib import Path
import json
import os

_CHATS_GOOD_PATH = Path(__file__).parent / "chats_good.json"
_CHATS_BAD_PATH = Path(__file__).parent / "chats_bad.json"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set")
    chats_good = json.loads(_CHATS_GOOD_PATH.read_text(encoding="utf-8"))
    chats_bad = json.loads(_CHATS_BAD_PATH.read_text(encoding="utf-8"))
    grades_good = grade_chats(chats_good)
    grades_bad = grade_chats(chats_bad)
    # Make sure the mean grades are below 0.5 for bad and above 0.5 for good
    bad_threshold, good_threshold = 0.5, 0.5
    assert grades_good.overall_mean_score > good_threshold
    assert grades_bad.overall_mean_score < bad_threshold
    print("✅ Tests passed")


if __name__ == "__main__":
    main()
