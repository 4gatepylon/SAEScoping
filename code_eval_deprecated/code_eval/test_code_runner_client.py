from __future__ import annotations
import math
import random
from typing import List, Tuple
import numpy as np

from utils.code_eval.code_runner_client import DockerizedTestRunner
from utils.code_eval.code_runner_schemas import (
    RunScriptBatchRequest,
    RunScriptBatchResponse,
)

"""
Integration test for DockerizedTestRunner that spins up multiple containers
and runs batch tests across them.
"""


def generate_template_script(sleep_time: float, operation: str) -> str:
    """Generate a script that adds two numbers with a random sleep."""
    return f"""
import time
import sys
import random

# Sleep for a random time to simulate work
time.sleep(random.uniform(0, {sleep_time}))

# Read two numbers from stdin
lines = sys.stdin.strip().split('\\n')
if len(lines) != 2:
    print("Error: Expected exactly 2 lines of input", file=sys.stderr)
    sys.exit(1)

try:
    a = float(lines[0])
    b = float(lines[1])
    result = a {operation} b
    print(result)
except ValueError as e:
    print(f"Error: Invalid number format - {{e}}", file=sys.stderr)
    sys.exit(1)
"""


def generate_addition_script(sleep_time: float) -> str:
    """Generate a script that adds two numbers with a random sleep."""
    return generate_template_script(sleep_time, "+")


def generate_multiplication_script(sleep_time: float) -> str:
    """Generate a script that multiplies two numbers with a random sleep."""
    return generate_template_script(sleep_time, "*")


def generate_test_cases(operation: str, num_cases: int = 128) -> List[Tuple[str, str]]:
    """Generate random test cases for the given operation."""
    a_vals = np.random.randint(low=0, high=100, size=num_cases)
    b_vals = np.random.randint(low=0, high=100, size=num_cases)
    results = (
        a_vals + b_vals
        if operation == "add"
        else a_vals * b_vals
        if operation == "multiply"
        else None
    )
    if results is None:
        raise ValueError(f"Unknown operation: {operation}")
    return [
        (f"{a}\\n{b}", str(result)) for a, b, result in zip(a_vals, b_vals, results)
    ]


class IntegrationTester:
    """Integration tests for the DockerizedTestRunner."""

    def __init__(
        self,
        n_containers: int = 7,  # number of containers to spin up
        n_scripts: int = 2,  # + and *
        n_test_cases: int = 128,
        min_sleep_time: float = 0.0,
        max_sleep_time: float = 1.0,  # timeout_per_run should get set to this
    ) -> None:
        self.n_containers = n_containers
        self.n_scripts = n_scripts
        self.n_test_cases = n_test_cases
        self.client = DockerizedTestRunner(
            docker_image="code-execution-server:latest",  # default image OK
            max_containers=n_containers,
        )
        self.min_sleep_time = min_sleep_time
        self.max_sleep_time = max_sleep_time

    def _create_batch_request(self, operation: str) -> RunScriptBatchRequest:
        """Create a batch request."""
        sleep_time = random.uniform(self.min_sleep_time, self.max_sleep_time)
        script = (
            generate_addition_script(sleep_time)
            if operation == "add"
            else generate_multiplication_script(sleep_time)
        )
        test_cases = generate_test_cases(operation, self.n_test_cases)
        return RunScriptBatchRequest(
            script_content=script,
            stdin_inputs=[tc[0] for tc in test_cases],
            # Expected outputs is set => Please grade
            expected_outputs=[tc[1] for tc in test_cases],
        )

    def _create_batch_requests(self) -> List[RunScriptBatchRequest]:
        """Create a list of batch requests."""
        batch_requests: List[RunScriptBatchRequest] = []
        for i in range(self.n_scripts):
            operation = "add" if i % 2 == 0 else "multiply"
            batch_requests.append(self._create_batch_request(operation))
        return batch_requests

    def test_dockerized_test_runner_client(self) -> None:
        """
        Test the runner client.

        Make sure that you are using large numbers if you want no timeouts.
        """
        batch_requests: List[RunScriptBatchRequest] = self._create_batch_requests()
        results: List[RunScriptBatchResponse] = self.client.run_batch_tests(
            batch_requests,
            timeout_per_run=self.max_sleep_time,
            # Note we tend to the mean for overall time, which should be
            # (
            #   0.5 *
            #   max_sleep_time *
            #   n_test_cases /
            #   n_scripts
            # )
            timeout_overall=math.ceil(  # ceil for safety
                0.7  # increase above mean
                * self.max_sleep_time  # max sleep time
                * self.n_test_cases  # number of test cases
                / self.n_scripts  # number of scripts
            ),
        )
        for i, result in enumerate(results):
            assert result.graded
            assert result.passed is not None
            assert result.pass_check_method is not None
            assert result.expected_outputs is not None

            # should have passed since the code is correct
            assert all(result.passed), (
                f"Script {i} had failures: {sum(not p for p in result.passed)} out of {len(result.passed)}"
            )
            assert len(result.results) == len(batch_requests[i].stdin_inputs)
            assert len(result.results) == len(result.passed)
            assert len(result.results) == len(result.expected_outputs)
            # Informative printouts
            print("=" * 100)
            print("Inputs:")
            print(" , ".join(batch_requests[i].stdin_inputs))
            print("\n\n")
            print("Expected outputs:")
            print(" , ".join(result.expected_outputs))
            print("\n\n")
            print("Results:")
            print(" , ".join([r.stdout for r in result.results]))
            print("=" * 100)


if __name__ == "__main__":
    integration_tester = IntegrationTester()
    integration_tester.test_dockerized_test_runner_client()
