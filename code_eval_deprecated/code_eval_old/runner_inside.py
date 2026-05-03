from __future__ import annotations
import traceback
import subprocess
import sys
from pathlib import Path
import os
import json  # should be in default library; fmt: skip
from typing import List, Any, Optional
import time

"""
This simple script is meant to be run inside a docker container. It will run
LLM-generated code and store the outputs in an output directory stipulated at
the top of the `main()` function.
"""


def main() -> None:
    time_start = time.time()
    input_test_generations_dir = Path("/input_test_generations")
    input_code_files_dir = Path("/input_files")
    output_generations_files_dir = Path("/output_generations_files")

    input_test_generations_files = list(input_test_generations_dir.iterdir())
    input_code_files = list(input_code_files_dir.iterdir())
    output_generations_files = list(output_generations_files_dir.iterdir())
    assert len(input_test_generations_files) == len(input_code_files)
    assert len(output_generations_files) == 0

    test_timeout = os.getenv("TEST_TIMEOUT", 10)
    test_timeout = float(test_timeout)
    overall_timeout = os.getenv("OVERALL_TIMEOUT", 100)
    overall_timeout = float(overall_timeout)
    assert overall_timeout >= test_timeout, f"Overall timeout must be greater than test timeout, got {overall_timeout} and {test_timeout}"  # fmt: skip
    print(f"Test_timeout = {test_timeout} sec")
    print("-" * 100)
    print("Found the following input_test_generations_files:")
    print("\n".join([x.name for x in input_test_generations_files]))
    print("-" * 100)
    print("Have the following input_code_files:")
    print("\n".join([x.name for x in input_code_files]))
    print("-" * 100)
    for i, input_test_generation_file in enumerate(input_test_generations_files):
        # 1. Fetch the object
        print("=" * 100)
        print(f"i={i}/{len(input_test_generations_files)} = {i/len(input_test_generations_files)} frac of way through")  # fmt: skip
        input_object = json.loads(input_test_generation_file.read_text())
        identifier = input_object["test_runtime_identifier"]
        assert identifier is not None

        # 2. Fetch the code file
        input_code_file = input_code_files_dir / f"{identifier}.py"
        assert input_code_file.exists()

        # 3. Run the python code file with standard input and output as the inputs and outputs
        inputs = input_object["inputs"]
        expected_outputs = input_object["expected_outputs"]
        assert inputs is not None and expected_outputs is not None
        assert len(inputs) == len(expected_outputs)
        actual_outputs: List[Any] = []
        passed: List[bool] = []
        errors: List[Optional[str]] = []
        for j, (_input, expected_output) in enumerate(zip(inputs, expected_outputs)):
            # Run that python file with the input and expected output
            # Implemented by Claude
            try:
                # Run the Python file with the input
                result = subprocess.run(
                    [sys.executable, input_code_file.resolve().as_posix()],
                    input=str(_input),  # Convert input to string for stdin
                    capture_output=True,
                    text=True,
                    timeout=float(test_timeout),  # 10 second timeout per test
                )
                if result.returncode != 0:
                    actual_outputs.append(None)  # No answer => None
                    errors.append(result.stderr.strip())
                    passed.append(False)

                else:
                    actual_output = result.stdout.strip()
                    actual_outputs.append(actual_output)
                    # Compare outputs (converting expected to string for comparison)
                    passed.append(
                        str(expected_output).strip() == str(actual_output).strip()
                    )
                    errors.append(None)  # No error => None
            except subprocess.TimeoutExpired:
                actual_outputs.append(None)
                passed.append(False)
                errors.append("ERROR: Timeout (inner subprocess)")

            except Exception as e:
                exc_msg = traceback.format_exc()
                msg = f"EXCEPTION:\n\n{e}" + "=" * 100 + f"\n\n{exc_msg}"
                actual_outputs.append(None)
                passed.append(False)
                errors.append(msg)
            finally:
                # Do this no matter what
                curr_time = time.time()
                # If we will run out of time assuming next one goes over
                d_time = curr_time - time_start
                if overall_timeout - d_time < test_timeout:
                    for _ in range(len(inputs) - j - 1):
                        actual_outputs.append(None)
                        passed.append(False)
                        errors.append(
                            "ERROR: Timeout (pre-empted outer process in inner loop)"
                        )
                    break  # exit inner for loop to go to next generations (those preempt)

        # 4. Update this test object
        assert len(actual_outputs) == len(expected_outputs)
        assert len(passed) == len(expected_outputs)
        assert len(errors) == len(expected_outputs)
        input_object["actual_outputs"] = actual_outputs
        input_object["passed"] = passed
        input_object["errors"] = errors

        output_file = output_generations_files_dir / input_test_generation_file.name
        assert not output_file.exists()
        output_file.write_text(json.dumps(input_object))

        # 5. pre-empt again
        curr_time = time.time()
        d_time = curr_time - time_start
        if curr_time - time_start > overall_timeout:
            for k in range(i + 1, len(input_test_generations_files)):
                k_output_object = json.loads(input_test_generations_files[k].read_text())  # fmt: skip
                k_output_file = output_generations_files_dir / input_test_generations_files[k].name  # fmt: skip
                assert not k_output_file.exists()
                k_output_object["actual_outputs"] = [None] * len(expected_outputs)
                k_output_object["passed"] = [False] * len(expected_outputs)
                k_output_object["errors"] = ["ERROR: Timeout (pre-empted outer process in outer loop)"] * len(expected_outputs)  # fmt: skip
                k_output_file.write_text(json.dumps(k_output_object))

    # by now we should have populated this stuff
    assert len(list(output_generations_files_dir.iterdir())) == len(input_test_generations_files)  # fmt: skip
    print("-" * 100)
    print("Produced the following output_generations_files:")
    print("\n".join([x.name for x in output_generations_files_dir.iterdir()]))
    print("-" * 100)


if __name__ == "__main__":
    main()
