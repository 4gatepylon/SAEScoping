from __future__ import annotations
import uuid
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils.code_eval_old.parse_utils import parse_generation_for_code
import orjson
import subprocess
from utils.code_eval_old.runner_lib import (
    TestGeneration,
    ParseArguments,
    PARSE_INFO_OPTIONS_MAP,
)
import click
import tqdm


class NoGenerationsError(Exception):
    pass  # lol


class TestRunner:
    """
    Provide functions to run multiple different tests on given code.

    The process is like this:
    1. LLM Generates generations to prompts (asking for code) and it gets outputed in a file
    somewhere in the format this script can read.
    -> We start here assuming (1) happened OK
    2. We load those files and then use `parse_utils.py` to extract the code from each
    generation.
    3. We then run the code in a container (or not) using the functions here.
    4. Using the methodology from APPS (https://github.com/hendrycks/apps) we measure both:
        - Per-problem results (keeping the metadata from teh generations; this way we can
            do stratified analysis later too; options are provided for some defaults below).
    5. Outputs of (4) are stored in JSON/Plots/etc... in the desired outputs directory.

    So generally, this gives you a one-click way to test a bunch of generations. It is meant
    for batched workflows, so you should generally just architect the rest of your stuff to
    merely generate the generations in the right format. Use tempfile as needed.

    NOTE: you will need permissions to run docker-compose (and docker). The program will
    not propagate sudo, so you must run it from a user that has those permissions.
    """

    def __init__(
        self,
        generations_file_folders: List[Path | str],
        # Default docker-compose.yaml is should work...
        docker_compose_yaml: Path | str = Path(__file__).parent / "docker-compose.yaml",
        # In a different subprocess we run docker-compose <args> <kwargs>
        docker_compose_args: List[str] = [
            "up",  # sync -> wait for it to finish
            "--build",  # force rebuild so we get latest version (no cached dockerfile)
        ],
        docker_compose_kwargs: Dict[str, Any] = {},
        # timeouts
        overall_test_timeout: float = 100.0,
        per_test_timeout: float = 10.0,
    ):
        # Generations stuff
        self.generations_file_folders: List[Path] = [
            Path(folder) for folder in generations_file_folders
        ]
        self.generations_files: Optional[List[Path]] = None
        self.generations: Optional[List[TestGeneration]] = None

        # Docker stuff
        self.docker_compose_yaml: Path = Path(docker_compose_yaml)
        if not self.docker_compose_yaml.exists():
            raise FileNotFoundError(
                f"Docker compose yaml file not found: {self.docker_compose_yaml}"
            )
        self.docker_compose_args: List[str] = docker_compose_args
        self.docker_compose_kwargs: Dict[str, Any] = docker_compose_kwargs

        # timeouts n management stuff
        self.overall_test_timeout: float = overall_test_timeout
        self.per_test_timeout: float = per_test_timeout

    ################ [BEGIN] Helpers for loading generations [BEGIN] ################
    def _get_generations_files_from_folder(
        self, generations_file_folder: Path, patterns=["*.json", "*.jsonl"]
    ) -> List[Path]:
        files = []
        for pattern in patterns:
            files.extend(list(generations_file_folder.glob(pattern)))
        return sorted(list(set(files)))

    def _get_generations_files_from_folders(
        self, generations_file_folders: List[Path]
    ) -> List[Path]:
        generations_files = []
        for generations_file_folder in generations_file_folders:
            generations_files.extend(
                self._get_generations_files_from_folder(generations_file_folder)
            )
        return generations_files

    def _parse_test_generations_files(
        self, generations_files: List[Path]
    ) -> List[TestGeneration]:
        """
        Return `TestGeneration` objects from the generations files that store them.
        """
        generations = []
        for generations_file in generations_files:
            generations.extend(
                TestGeneration.parse_test_generations_file(generations_file)
            )
        for i in range(len(generations)):
            if generations[i].code is None:
                parse_info = ParseArguments()  # Defaults
                if generations[i].parse_info is None:
                    raise ValueError(
                        "Parse info is None but no code is provided"
                    )  # Unreachable; fmt: skip
                elif generations[i].parse_info in PARSE_INFO_OPTIONS_MAP:
                    parse_info = PARSE_INFO_OPTIONS_MAP[generations[i].parse_info]
                elif isinstance(generations[i].parse_info, str):
                    raise ValueError(f"Invalid parse info: {generations[i].parse_info}")
                else:
                    parse_info = generations[i].parse_info
                generations[i].code = parse_generation_for_code(
                    generations[i].generation,
                    func_name=parse_info.func_name,
                    func_kwargs=parse_info.func_kwargs,
                )
        assert all(generations[i].code is not None for i in range(len(generations))), "All generations must have code"  # fmt: skip
        return generations

    ################ [END] Helpers for loading generations [END] ################

    def load_generations(self) -> None:
        self.generations_files = self._get_generations_files_from_folders(
            self.generations_file_folders
        )
        self.generations = self._parse_test_generations_files(self.generations_files)
        # Make sure to populate the identifier (this will be used to match the
        # inputs/outputs to specific python code files that should be run)
        self.generations = [
            generation.model_copy(update={"test_runtime_identifier": str(uuid.uuid4())})
            for generation in self.generations
        ]
        if len(self.generations) == 0:
            raise NoGenerationsError("No generations loaded... not supported!")

    def loaded_generations(self) -> bool:
        assert (self.generations_files is None) == (self.generations is None)
        return self.generations_files is not None

    def run_docker_compose(self) -> List[TestGeneration]:
        """
        Run `runner_inside.py` inside a docker container, passing inputs and
        outputs via folders attached as volumes.
        """
        # 1. Populate the inputs
        assert self.generations is not None
        assert all(g.test_runtime_identifier is not None for g in self.generations), "All generations must have a test_runtime_identifier"  # fmt: skip
        assert all(g.code is not None for g in self.generations), "All generations must have code"  # fmt: skip
        with tempfile.TemporaryDirectory() as _temp_dir:
            input_dir = Path(_temp_dir) / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir = Path(_temp_dir) / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            code_dir = Path(_temp_dir) / "code"
            code_dir.mkdir(parents=True, exist_ok=True)
            for generation in self.generations:
                assert generation.test_runtime_identifier is not None
                input_file = input_dir / f"{generation.test_runtime_identifier}.json"
                assert not input_file.exists()  # write once
                input_file.write_text(generation.model_dump_json())
                code_file = code_dir / f"{generation.test_runtime_identifier}.py"
                assert not code_file.exists()  # write once
                assert generation.code is not None
                code_file.write_text(generation.code)
            assert len(list(input_dir.iterdir())) == len(self.generations)
            assert len(list(code_dir.iterdir())) == len(self.generations)

            # Set up environment variables for docker-compose
            env4docker_compose = os.environ.copy()
            env4docker_compose.update(
                {
                    "INPUT_TEST_GENERATIONS_DIR": input_dir.resolve().as_posix(),
                    "INPUT_CODE_FILES_DIR": code_dir.resolve().as_posix(),
                    "OUTPUT_GENERATIONS_FILES_DIR": output_dir.resolve().as_posix(),
                    "TEST_TIMEOUT": str(self.per_test_timeout),
                    "OVERALL_TIMEOUT": str(self.overall_test_timeout),
                }
            )

            # 2. Run docker-compose
            docker_compose_cmd = [
                "docker-compose",
                "-f",
                self.docker_compose_yaml.resolve().as_posix(),
            ]
            docker_compose_cmd.extend(self.docker_compose_args)

            try:
                result = subprocess.run(
                    docker_compose_cmd,
                    env=env4docker_compose,
                    check=True,
                    # capture_output=True, # Use if you prefer not to printout/stream
                    text=True,
                    cwd=self.docker_compose_yaml.parent,  # Run from the directory containing docker-compose.yaml
                    timeout=self.overall_test_timeout,
                )
            except subprocess.CalledProcessError as e:
                print("=" * 100)
                print(f"Docker compose failed with return code {e.returncode}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                print("=" * 100)
                raise

            # 3. Extract the results
            output_files = list(output_dir.iterdir())
            assert len(output_files) == len(self.generations)
            results = [
                orjson.loads(output_file.read_bytes()) for output_file in output_files
            ]

        # Return once folder is cleaned up
        return [TestGeneration.model_validate(result) for result in results]

    def run_tests(self) -> List[TestGeneration]:
        """
        To do this:
        1. Load the generations (into memory, now we have the code that we need to test)
        2. Populate the app folder (basically fill it up with the code to run the tests)
        3. Build the docker container by using docker-compose <args> <kwargs> with
            another subprocess.
        4. Run the docker container (this just invokes a default run-script that is
            copied into the docker container).
        5. Parse & return/store the results. Synchronous for now.
        """
        # 1. Load the generations
        self.load_generations()
        assert self.loaded_generations()
        # 2. Populate the app folder + run docker + extract results
        results = self.run_docker_compose()
        assert len(results) == len(self.generations)
        # All results must either have actual_outputs or errors
        assert all((result.actual_outputs is not None) or (result.errors is not None) for result in results), "All results must have either actual_outputs or errors"  # fmt: skip
        # All results should have passed (false if errors)
        assert all((result.passed is not None) for result in results if result.errors is None), "All results must have passed if errors is None"  # fmt: skip
        # all these arrays same length
        assert all(
            (
                (
                    len(result.actual_outputs)
                    if result.actual_outputs is not None
                    else len(result.expected_outputs)
                )  # fmt: skip
                == len(result.expected_outputs)
                == len(result.inputs)
                == len(result.passed)
                == (
                    len(result.errors)
                    if result.errors is not None
                    else len(result.expected_outputs)
                )  # fmt: skip
            )
            for result in results
        )
        return results


@click.command()
@click.option(
    "--generations-file-folders",
    "-i",
    type=str,
    multiple=True,
    help="Folders containing generations files",
)
@click.option("--output-dir", "-o", type=str, help="Directory to store outputs")
def main(generations_file_folders: List[str | Path], output_dir: str) -> None:
    if output_dir is None or Path(output_dir).exists():
        raise ValueError("Output directory must be provided and not already exist")
    _output_dir = Path(output_dir)
    _output_dir.mkdir(parents=True, exist_ok=True)
    # Run tests
    runner = TestRunner(generations_file_folders=generations_file_folders)
    results = runner.run_tests()
    # Store the results
    assert all(result.test_runtime_identifier is not None for result in results), "All results must have a test_runtime_identifier"  # fmt: skip
    for result in tqdm.tqdm(results, desc="Storing results"):
        output_file = _output_dir / f"{result.test_runtime_identifier}.json"
        assert not output_file.exists()
        output_file.write_text(result.model_dump_json())
    print(f"Results stored in {_output_dir}")


if __name__ == "__main__":
    main()
