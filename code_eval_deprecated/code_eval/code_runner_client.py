from __future__ import annotations


import random
import time
from pathlib import Path
from typing import List, Optional, Type, Literal, Tuple, Any, Dict
import requests
import click
import docker
import pydantic
import multiprocessing
import tqdm
from utils.code_eval.code_runner_schemas import (
    # Server-side
    PostScriptRequest,
    PostScriptResponse,
    RunScriptRequest,
    ListScriptsRequest,
    GetScriptRequest,
    DeleteScriptRequest,
    ListScriptsResponse,
    GetScriptResponse,
    DeleteScriptResponse,
    RunScriptResponse,
    # Client-side
    RunScriptBatchRequest,
    RunScriptBatchResponse,
)

"""
Client for the code execution server. Supports both connecting to existing servers
and spinning up dockerized servers for batch processing.

By Claude. XXX review this plz
"""


class TestArgument(pydantic.BaseModel):
    """
    A single argument to a test request. This is a sorta-private class for this sole
    purpose.
    """

    # Send this to the server
    stdin_input: str

    # Recieve this and store only once recieved from server:
    response_object: Optional[RunScriptResponse] = None

    # Use this stuff to find stuff
    script_index: int  # used to index
    script_argument_index: int  # used to index

    def get_script_id(
        self,
        client_index: int,
        test_request_script_ids: List[List[str]],
    ) -> str:
        """Needed to send the actual request..."""
        # the script_argument index is whatever, it's used to sort inside the
        # responses that should be 1:1 with the script requests
        return test_request_script_ids[self.script_index][client_index]


class CodeExecutionClient:
    """Client for communicating with the code execution server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    ################ [BEGIN] Claude Boilerplate [BEGIN] #################
    def _http(
        self,
        request_object: pydantic.BaseModel,
        path: str,
        pydantic_class: Type[pydantic.BaseModel],
        method: Literal["POST", "GET", "DELETE"] = "POST",
    ) -> pydantic.BaseModel:
        """Make an HTTP request to the server."""
        response = self.session.post(
            f"{self.base_url}/{path}",
            json=request_object.model_dump(),
        )
        response.raise_for_status()
        json_response = response.json()
        return pydantic_class.model_validate(json_response)

    def post_script(self, request: PostScriptRequest) -> PostScriptResponse:
        """Upload a script to the server and return its ID."""
        return self._http(request, "post_script", PostScriptResponse)

    def delete_script(self, request: DeleteScriptRequest) -> DeleteScriptResponse:
        """Delete a script from the server."""
        return self._http(request, "delete_script", DeleteScriptResponse)

    def list_scripts(self, request: ListScriptsRequest) -> ListScriptsResponse:
        """List all scripts on the server."""
        return self._http(request, "list_scripts", ListScriptsResponse)

    def get_script(self, request: GetScriptRequest) -> GetScriptResponse:
        """Get a script by ID."""
        return self._http(request, "get_script", GetScriptResponse)

    def run_script(self, request: RunScriptRequest) -> RunScriptResponse:
        """Run a script with the given stdin input."""
        return self._http(request, "run_script", RunScriptResponse)

    ################ [END] Claude Boilerplate [END] #################


class DockerizedTestRunner:
    """
    Manages dockerized code execution servers for batch processing.

    TODO(Adriano) in the future support generic docker containers/images. Right now this
    is forced to work with the code-execution-server image because of `run_batch_tests`
    primarily.


    TODO(Adriano) in the future support other ways of calling this.

    This, right now, is meant to be run from a CLI basically.
    """

    def __init__(
        self,
        # TODO(Adriano) support parameterizing this
        docker_image: str = "code-execution-server:latest",
        max_containers: int = 4,
    ):
        self.docker_image = docker_image
        self.max_containers = max_containers
        self.docker_client = docker.from_env()
        self.active_containers: List[docker.models.containers.Container] = []

    def build_image(self, dockerfile_path: Path, context_path: Path) -> None:
        """Build the Docker image."""
        click.echo(f"Building Docker image {self.docker_image}...")
        self.docker_client.images.build(
            path=context_path.resolve().as_posix(),
            dockerfile=dockerfile_path.resolve().as_posix(),
            tag=self.docker_image,
            rm=True,
        )
        click.echo("Docker image built successfully!")

    def start_container(
        self,
        port: int = 8000,
        n_tries: int = 30,
        wait_before_starting: float = 1.0,
        sleep_between_tries: float = 1.0,
    ) -> Tuple[docker.models.containers.Container, CodeExecutionClient]:
        """Start a single container on the specified port."""
        # Count number of active containers:
        n_active_containers_before = len(self.docker_client.containers.list())

        container = self.docker_client.containers.run(
            self.docker_image,
            detach=True,
            ports={f"{port}/tcp": port},
            remove=True,
            name=f"code-exec-{port}",
        )
        time.sleep(wait_before_starting)

        # Check that the docker container exists by looking at docker list
        n_active_containers_after = len(self.docker_client.containers.list())
        if n_active_containers_after <= n_active_containers_before:
            raise RuntimeError(f"Container on port {port} failed to start")

        # Wait for container to be ready
        client = CodeExecutionClient(f"http://localhost:{port}")
        for _ in tqdm.trange(n_tries, desc="Waiting for container to be ready"):
            try:
                response = requests.get(
                    f"http://localhost:{port}/list_scripts", timeout=1
                )
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(sleep_between_tries)
        else:
            container.stop()
            raise RuntimeError(f"Container on port {port} failed to start")

        return container, client

    def start_containers(
        self,
        must_start_at_least: int = 0,
        base_port: int = 8000,
    ) -> List[CodeExecutionClient]:
        """Start multiple containers and return clients for them."""
        clients = []
        ports = []
        for i in range(self.max_containers):
            port = base_port + i
            try:
                container, client = self.start_container(port)
                self.active_containers.append(container)
                clients.append(client)
                ports.append(port)
            except Exception as e:
                click.echo(f"Failed to start container on port {port}: {e}")
                break

        if not clients:
            raise RuntimeError("Failed to start any containers")

        click.echo(f"Started {len(clients)} containers")
        if len(clients) < must_start_at_least:
            # Delete all then raise
            for container in self.active_containers:
                container.stop()
            self.active_containers.clear()
            raise RuntimeError(f"Failed to start {must_start_at_least} containers")
        return clients

    def stop_containers(self) -> None:
        """Stop all active containers."""
        for container in self.active_containers:
            try:
                container.stop()
            except Exception as e:
                click.echo(f"Error stopping container {container.name}: {e}")

        self.active_containers.clear()

    @staticmethod
    def _worker_process(
        # self, # Static cuz docker client cannot be pickled :/
        arguments: Tuple[
            List[List[str]],  # test_request_script_ids
            int,  # client_idx
            str,  # base_url
            List[Dict[str, Any]],  # test_arguments (serialized)
            Optional[float],  # timeout_overall
            Optional[float],  # timeout_per_run
        ],
    ) -> List[Dict[str, Any]]:
        """
        A worker process that runs a single test request.
        """
        (
            test_request_script_ids,
            client_idx,
            base_url,
            serialized_test_arguments,
            timeout_overall,
            timeout_per_run,
        ) = arguments

        test_arguments: List[TestArgument] = [
            TestArgument.model_validate(serialized_test_argument)
            for serialized_test_argument in serialized_test_arguments
        ]

        if timeout_overall is not None:
            raise NotImplementedError(
                "Not implemented. Timeout overall requires further processing"
            )

        # TODO(Adriano) client is not actually stateful tbh (other than for session, so
        # this usage is fine) but we should clean this up so we don't have useless dangling client objects
        client = CodeExecutionClient(base_url)
        for test_argument in test_arguments:
            script_id = test_argument.get_script_id(client_idx, test_request_script_ids)

            # There should be no errors (server should handle this...)
            response = client.run_script(
                RunScriptRequest(
                    script_id=script_id,
                    stdin_input=test_argument.stdin_input,
                    timeout=timeout_per_run,
                )
            )
            test_argument.response_object = response

        # Return serialized test arguments with responses
        return [test_arg.model_dump() for test_arg in test_arguments]

    def _grade_results(
        self, results: List[RunScriptBatchResponse]
    ) -> List[RunScriptBatchResponse]:
        """Grade the results based on expected outputs and pass_check_method."""
        for result in results:
            if not result.graded or result.expected_outputs is None:
                continue

            # Ensure we have the same number of results and expected outputs
            if len(result.results) != len(result.expected_outputs):
                click.echo(
                    f"Warning: Mismatch in number of results ({len(result.results)}) and expected outputs ({len(result.expected_outputs)})"
                )
                continue

            # Grade each result
            passed = []
            for actual_result, expected_output in zip(
                result.results, result.expected_outputs
            ):
                actual_output = actual_result.stdout

                if result.pass_check_method == "exact_match":
                    is_passed = actual_output == expected_output
                elif result.pass_check_method == "exact_match_with_strip":
                    is_passed = actual_output.strip() == expected_output.strip()
                else:
                    raise NotImplementedError(
                        f"Pass check method {result.pass_check_method} not implemented"
                    )

                passed.append(is_passed)

            result.passed = passed

        return results

    def run_batch_tests(
        self,
        test_requests: List[RunScriptBatchRequest],
        timeout_overall: Optional[float] = None,
        timeout_per_run: Optional[float] = None,
        must_start_at_least: int = 1,
        pass_check_method: Literal[
            "exact_match", "exact_match_with_strip"
        ] = "exact_match_with_strip",
    ) -> List[RunScriptBatchResponse]:
        """
        Run a batch of test requests across multiple containers.

        This load-balances using a very stupid strategy. Basically, it first uploads ALL
        python scripts to ALL servers (all containers). Then, it evenly distributes ALL
        the test requests across ALL servers.

        The way this is run in parallel is simple:
        1. We start up all servers
        2. We upload all scripts to all servers (serially; this should be fast since we
            assume there are few scripts but many test-cases)
        3. We create one list of inputs to send to the server per server.
        4. We run one subprocess per server to just feed into the server. When this
            subprocess finishes it returns. Our master process runs a process pool of
            these subprocesses and waits for them to ALL finish. Then, it concatenates
            the results. To maintain order, we send both the index and the actual
            argument.
        5. We serially do grading in the end if necessary.

        To get a little bit of a boost we also randomly shuffle the test-cases before
        distributing them (this is why we need to keep track of indices).

        One implication is that requests will be sent in a "flat" manner, such that
        across different RunScriptBatchRequests the actual requests are mixed. Thus, at
        the end both the index in the list of RunScriptBatchRequests and the index in
        the list of request arguments INSIDE That are used to index the results.
        """
        # check if docker image exists
        if not self.docker_client.images.list(name=self.docker_image):
            print("=" * 100)
            print("Docker image does not exist, building image...")
            print("=" * 100)
            self.build_image(
                Path(__file__).parent / "Dockerfile",
                Path(__file__).parent,
            )
            assert len(self.docker_client.images.list(name=self.docker_image)) >= 1

        clients = self.start_containers(must_start_at_least)

        # test_request_index2client_index2script_id
        test_request_script_ids: List[List[str]] = []
        results = []

        # 1. Upload all scripts to all servers
        for i, test_request in enumerate(test_requests):
            # 1.1 Upload to all servers
            client_script_ids: List[str] = []
            for client in tqdm.tqdm(clients, desc="Uploading scripts"):
                response: PostScriptResponse = client.post_script(
                    PostScriptRequest(
                        script_content=test_request.script_content,
                        filename=f"script_{i}.py",
                        metadata=None,
                    )
                )
                client_script_ids.append(response.script_id)

            # 1.2 Ensure that we know script ids to use for the specific TestArguments
            assert len(client_script_ids) == len(clients)
            test_request_script_ids.append(client_script_ids)

            # 1.3 Update results so that it matches 1:1
            results.append(
                RunScriptBatchResponse(
                    results=[],  # gonna be appended l8r (and then sorted)
                    graded=test_request.expected_outputs is not None,
                    # stuff that will be set l8r during grading phase
                    expected_outputs=test_request.expected_outputs,
                    pass_check_method=pass_check_method,
                    passed=[False] * len(test_request.stdin_inputs),
                )
            )
        assert len(test_request_script_ids) == len(test_requests)
        assert len(results) == len(test_requests)

        # 2. Create all arguments and shuffle them so that we can distribute them evenly
        all_arguments: List[TestArgument] = []
        for i, test_request in enumerate(
            tqdm.tqdm(test_requests, desc="Creating arguments")
        ):
            for j, stdin_input in enumerate(test_request.stdin_inputs):
                all_arguments.append(
                    TestArgument(
                        script_index=i,
                        script_argument_index=j,
                        stdin_input=stdin_input,  # for server
                    )
                )
        random.shuffle(all_arguments)

        # 3. Evenly distribute arguments and then launch parallel workers to get the
        # results (and create something that can go over the wire)
        worker_args = [
            (
                test_request_script_ids,
                client_idx,
                client.base_url,
                list(map(lambda x: x.model_dump(), all_arguments[i :: len(clients)])),
                timeout_overall,
                timeout_per_run,
            )
            for client_idx, client in enumerate(clients)
        ]
        try:
            with multiprocessing.Pool(processes=len(clients)) as pool:
                worker_results = pool.map(
                    DockerizedTestRunner._worker_process,
                    worker_args,
                )
            all_processed_result_test_arguments = [
                TestArgument.model_validate(result)
                for worker_result in worker_results
                for result in worker_result
            ]
            for i, result_test_argument in enumerate(
                all_processed_result_test_arguments
            ):
                assert result_test_argument.response_object is not None
                # scripts 1:1 with test_requests 1:1 with results
                results_index = result_test_argument.script_index
                # results_argument_index = result_test_argument.script_argument_index # used later for sort; fmt: skip
                results[results_index].results.append(
                    result_test_argument.response_object
                )
            # Make sure all have the right length
            assert len(results) == len(test_requests)
            assert all(len(results[i].results) == len(test_requests[i].stdin_inputs) for i in range(len(results)))  # fmt: skip
            # Make sure these are all properly in order and check the indices
            for r in tqdm.tqdm(results, desc="Sorting results"):
                r.results = sorted(
                    r.results,
                    key=lambda x: x.script_argument_index,
                )
                assert list(range(len(r.results))) == [x.script_argument_index for x in r.results]  # fmt: skip

        finally:
            self.stop_containers()

        # 4. Grade the results (this will not grade if not necessary)
        results = self._grade_results(results)

        return results
