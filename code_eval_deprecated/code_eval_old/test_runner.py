from __future__ import annotations


from typing import List
import tempfile
from pathlib import Path
from utils.code_eval_old.runner_lib import TestGeneration
from utils.code_eval_old.runner_outside import TestRunner

"""
Integration test for both `runner_inside.py` and `runner_outside.py`. This uses a
single dummy test code file.
"""


def main() -> None:
    ################ [BEGIN] Define the code and tests [BEGIN] ################
    code = """def add(a, b):
    return a + b
# read stdin
input = input()
try:
    a, b = map(int, input.split())
    print(add(a, b))
except Exception as e:
    print(-1)
"""
    tests = [
        ("1 2", "3"),
        ("3 4", "7"),
        ("5 6", "11"),
        ("7 8", "15"),
        ("9 10", "19"),
        ("asdfasdf", "-1"),
        ("1 2 3", "-1"),
        ("-1 -2", "-3"),
        ("-1 1", "0"),
        ("0 0", "0"),
        ("1 ", "-1"),
        ("1", "-1"),
        ("-1 0", "-1"),
    ]
    ################ [END] Define the code and tests [END] ################
    #
    # Run a simple integration test with dummy code, this should be able to demonstrate
    # that our thing runs at all
    generations: List[TestGeneration] = [
        TestGeneration(
            # No need to parse
            generation=None,
            parse_info=None,
            # Code is what will be extracted
            code=code,
            # Inputs specify correct behavior
            inputs=[i for i, _ in tests],
            expected_outputs=[o for _, o in tests],
            # These should be populated by both runners
            actual_outputs=None,
            passed=None,
            errors=None,
            test_runtime_identifier=None,
            # Metadata is unused for this test
            metadata={"a": "b"},  # dummy
        )
    ]
    with tempfile.TemporaryDirectory() as _temp_dir:
        # 1. Define the generations and save them to a folder we will load from
        temp_dir = Path(_temp_dir)
        test_generations_dir = temp_dir / "test_generations"
        test_generations_dir.mkdir(parents=True, exist_ok=True)
        test_generations_file = test_generations_dir / "test_generations.json"
        test_generations_file.write_text(generations[0].model_dump_json())
        # 2. Create the tester objects and run tests
        tester = TestRunner(generations_file_folders=[test_generations_dir])
        results = tester.run_tests()
        # 3. Print the results
        print("=" * 100)
        print(f"len(results)={len(results)}")
        assert len(results) == 1
        print("Results:")
        print("=" * 100)
        print(results[0].model_dump_json(indent=4))


if __name__ == "__main__":
    main()
