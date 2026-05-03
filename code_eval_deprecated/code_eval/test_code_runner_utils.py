from __future__ import annotations
import tqdm
import pytest
from utils.code_eval.code_runner_utils import ScriptsHandler
from utils.code_eval.code_runner_schemas import (
    PostScriptRequest,
    RunScriptRequest,
    ListScriptsRequest,
    GetScriptRequest,
    DeleteScriptRequest,
)


class TestScriptsHandler:
    """
    A pydantic tester for scripts handler.

    Mostly, by Claude.

    TODO(Adriano) decide what other edge-cases we want to test. We should also probably
        test the following (but tbd):
    - Passing whitespace does not get stripped (or does it? what do we want?)
    - Things work OK if the script opens files or does other shenanigans (we probably
        want docker or some way of mocking the FS to avoid breaking stuff)
    - Things work if the script launches subprocesses, threads, etc...
    - Unicode works OK
    - Filename repeated is OK
    - Access the scripts concurrently (so one object, multiple threads manipulate it;
        I'm frankly not sure what we want here)
    - Potentially something to do with compute being too much? (again, not sure what we
        want here and we need to mock out the computational units to not brick the host)
    - Define clearly how precise the timeout is
    - More clearly/defitively check the contents of the stdout/stderr
    - Check determinism, especially when parsing outputs for newlines and other stuff
        that may be cut off.

    (possibly define what we want the timeout behavior to be here... --- I think you
    cannot override default right?)
    """

    def test_running_trivial_script(self) -> None:
        # 1. Define script (behavior is obvious)
        script = """
_input = input()
print("Hello, " + str(_input) + "!")
"""
        # 2. Create handler
        handler = ScriptsHandler()
        # 3. Upload script
        request = PostScriptRequest(script_content=script, filename="test.py")
        response = handler.post_script(request)
        script_id = response.script_id

        # 4. Run script + check for no errors
        request = RunScriptRequest(script_id=script_id, stdin_input="world")
        result = handler.run_script(request)
        assert result.stdout == "Hello, world!\n"
        assert result.return_code == 0
        assert result.time_taken_seconds >= 0

    def test_posting_getting_and_listing_scripts(self) -> None:
        """
        Test posting multiple scripts, listing them, and getting individual scripts.
        (make sure that metadata is there IFF you request, that the contents are correct
        etc...)
        """
        # 1. Create handler
        handler = ScriptsHandler()

        # 2. Post multiple scripts with different metadata
        scripts_data = [
            {
                "content": 'print("Script 1")',
                "filename": "script1.py",
                "metadata": {"author": "Alice", "version": 1},
            },
            {
                "content": 'print("Script 2")\nprint("Line 2")',
                "filename": "script2.py",
                "metadata": {"author": "Bob", "tags": ["test", "demo"]},
            },
            {
                "content": 'x = 5\nprint(f"x = {x}")',
                "filename": "script3.py",
                "metadata": None,  # No metadata
            },
        ]

        script_ids = []
        for data in scripts_data:
            request = PostScriptRequest(
                script_content=data["content"],
                filename=data["filename"],
                metadata=data["metadata"],
            )
            response = handler.post_script(request)
            script_ids.append(response.script_id)

        # 3. List scripts with metadata included
        list_request = ListScriptsRequest(include_metadata=True)
        list_response = handler.list_scripts(list_request)

        # Verify all scripts are present
        assert len(list_response.scripts) == 3

        # Verify script_content is always None in list
        for script_info in list_response.scripts:
            assert script_info.script_content is None

        # Verify metadata is included when requested
        script_id_to_metadata = {
            sid: data["metadata"] for sid, data in zip(script_ids, scripts_data)
        }
        for script_info in list_response.scripts:
            assert script_info.script_id in script_id_to_metadata  # incl. all scripts
            expected_metadata = script_id_to_metadata[script_info.script_id]
            if expected_metadata is None:
                assert script_info.has_metadata is False
                assert script_info.metadata is None
            else:
                assert script_info.has_metadata is True
                assert script_info.metadata == expected_metadata

        # 4. List scripts without metadata
        list_request_no_meta = ListScriptsRequest(include_metadata=False)
        list_response_no_meta = handler.list_scripts(list_request_no_meta)

        # Verify metadata is None when not requested
        for script_info in list_response_no_meta.scripts:
            assert script_info.metadata is None
            # But has_metadata should still indicate if metadata exists
            if script_info.script_id == script_ids[2]:  # Script 3 has no metadata
                assert script_info.has_metadata is False
            else:
                assert script_info.has_metadata is True

        # 5. Get individual scripts and verify content
        for i, (script_id, data) in enumerate(zip(script_ids, scripts_data)):
            get_request = GetScriptRequest(script_id=script_id, include_metadata=True)
            get_response = handler.get_script(get_request)

            assert get_response.script_id == script_id
            assert get_response.filename == data["filename"]
            assert get_response.script_content == data["content"]
            assert get_response.size_bytes == len(data["content"].encode("utf-8"))

            if data["metadata"] is None:
                assert get_response.has_metadata is False
                assert get_response.metadata is None
            else:
                assert get_response.has_metadata is True
                assert get_response.metadata == data["metadata"]

    def test_running_and_deleting_trivial_scripts(self) -> None:
        """Test running a script and then deleting it."""
        # 1. Create handler
        handler = ScriptsHandler()

        # 2. Post a simple script
        script_content = """
import sys
name = input("Enter your name: ")
print(f"Hello, {name}!")
print(f"Python version: {sys.version.split()[0]}")
"""
        post_request = PostScriptRequest(
            script_content=script_content,
            filename="greeting.py",
            metadata={"purpose": "greeting"},
        )
        post_response = handler.post_script(post_request)
        script_id = post_response.script_id

        # 3. Run the script
        run_request = RunScriptRequest(script_id=script_id, stdin_input="Claude")
        run_response = handler.run_script(run_request)

        assert run_response.return_code == 0
        assert "Hello, Claude!" in run_response.stdout
        assert "Python version:" in run_response.stdout
        assert run_response.stderr == ""
        assert run_response.timed_out is False

        # 4. Verify script exists in list
        list_response = handler.list_scripts(ListScriptsRequest())
        assert any(s.script_id == script_id for s in list_response.scripts)

        # 5. Delete the script
        delete_request = DeleteScriptRequest(script_id=script_id)
        delete_response = handler.delete_script(delete_request)

        assert delete_response.success is True
        assert delete_response.exists is True

        # 6. Verify script no longer exists in list
        list_response_after = handler.list_scripts(ListScriptsRequest())
        assert not any(s.script_id == script_id for s in list_response_after.scripts)

        # TODO(Adriano) the choice of 404 was by claude (which is fine by me for now,
        # but if this were a real server you probably prefer 500)
        # 7. Try to get the deleted script (should raise 404)
        with pytest.raises(Exception) as exc_info:
            handler.get_script(GetScriptRequest(script_id=script_id))
        assert (
            "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        )

        # 8. Try to run the deleted script (should raise 404)
        with pytest.raises(Exception) as exc_info:
            handler.run_script(RunScriptRequest(script_id=script_id))
        assert (
            "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        )

        # 9. Try to delete non-existent script
        delete_response_nonexistent = handler.delete_script(
            DeleteScriptRequest(script_id="fake-id")
        )
        assert delete_response_nonexistent.success is False
        assert delete_response_nonexistent.exists is False

    def test_running_script_that_throws_error(self) -> None:
        """Test running a script that raises an exception."""
        # 1. Create handler
        handler = ScriptsHandler()

        # 2. Post scripts with different error types
        error_scripts = [
            {
                "content": """
# Division by zero error
x = 10
y = 0
result = x / y
print(f"Result: {result}")
""",
                "filename": "divide_by_zero.py",
                "error_type": "ZeroDivisionError",
            },
            {
                "content": """
# Name error
print(f"Value: {undefined_variable}")
""",
                "filename": "name_error.py",
                "error_type": "NameError",
            },
            {
                "content": """
# Syntax error - this is different as it fails at parse time
print("Before error")
if True
    print("This has syntax error")
""",
                "filename": "syntax_error.py",
                "error_type": "SyntaxError",
            },
            {
                "content": """
# Custom exception with partial output
print("Starting process...")
print("Step 1 complete")
raise RuntimeError("Something went wrong!")
print("This won't print")
""",
                "filename": "runtime_error.py",
                "error_type": "RuntimeError",
                "metadata": {
                    "check_stdout": True,
                    "check_stdout_contains": ["Starting process...", "Step 1 complete"],
                    "check_stdout_does_not_contain": ["This won't print"],
                },
            },
            # Malicious inputs cases
            {
                # Index error due to accessing an element that doesn't exist
                # Stdin should be passed as something less than 10 elements
                "content": """
my_list = list(map(int, input().strip().split()))
print(my_list[10])
""",
                "filename": "index_error.py",
                "error_type": "IndexError",
                "stdin_input": "1 2 3 4 5 6",
            },
            {
                # Value error due to mapping non-integer
                "content": """
my_list = list(map(int, input().strip().split()))
print(my_list[10])
""",
                "filename": "value_error.py",
                "error_type": "ValueError",
                "stdin_input": "1 2 3 4 5 6 7 8 9 10 11 12 hello",
            },
        ]

        for script_data in tqdm.tqdm(error_scripts, desc="Running error scripts"):
            # Post the script
            post_request = PostScriptRequest(
                script_content=script_data["content"], filename=script_data["filename"]
            )
            post_response = handler.post_script(post_request)

            # Run the script
            run_request = RunScriptRequest(
                script_id=post_response.script_id,
                stdin_input=script_data.get("stdin_input", ""),
            )
            run_response = handler.run_script(run_request)

            # Verify error behavior
            assert run_response.return_code != 0  # Non-zero exit code
            assert run_response.timed_out is False
            assert script_data["error_type"] in run_response.stderr

            # For runtime error, verify partial output was captured
            if script_data.get("metadata", {}).get("check_stdout", False):
                for check in script_data["metadata"]["check_stdout_contains"]:
                    assert check in run_response.stdout, f"Expected {check} in stdout"
                for check in script_data["metadata"]["check_stdout_does_not_contain"]:
                    assert check not in run_response.stdout, (
                        f"Did not expect {check} in stdout"
                    )

    def test_running_script_that_times_out(self) -> None:
        """
        Test running a script that exceeds the timeout limit.

        NOTE: this can take up to 10 seconds to run (because of the sleep times).
        """
        # 1. Create handler with default timeout
        handler = ScriptsHandler(script_timeout=5.0)  # Default 5 seconds

        # 2. Post a script that sleeps
        timeout_script = """
import time
print("Starting long operation...")
print(flush=True)  # Ensure output is flushed
time.sleep(3.0)  # Sleep for 3 seconds
print("Operation complete!")
"""
        post_request = PostScriptRequest(
            script_content=timeout_script, filename="slow_script.py"
        )
        post_response = handler.post_script(post_request)
        script_id = post_response.script_id

        # 3. Run with sufficient timeout (should complete)
        run_request_ok = RunScriptRequest(script_id=script_id, timeout=4.0)
        run_response_ok = handler.run_script(run_request_ok)

        assert run_response_ok.return_code == 0
        assert run_response_ok.timed_out is False
        assert "Starting long operation..." in run_response_ok.stdout
        assert "Operation complete!" in run_response_ok.stdout
        assert run_response_ok.timeout_time_seconds == 4.0

        # 4. Run with insufficient timeout (should timeout)
        run_request_timeout = RunScriptRequest(script_id=script_id, timeout=0.5)
        run_response_timeout = handler.run_script(run_request_timeout)

        # Return code and stdout/stderr carry no guarantees here
        assert run_response_timeout.timed_out is True
        assert run_response_timeout.timeout_time_seconds == 0.5
        assert run_response_timeout.error_message is not None

        # 5. Test with infinite loop to ensure timeout works
        infinite_script = """
print("Starting infinite loop...")
while True:
    pass  # Infinite loop
print("This will never print")
"""
        post_request_inf = PostScriptRequest(
            script_content=infinite_script, filename="infinite_loop.py"
        )
        post_response_inf = handler.post_script(post_request_inf)

        run_request_inf = RunScriptRequest(
            script_id=post_response_inf.script_id, timeout=1.0
        )
        run_response_inf = handler.run_script(run_request_inf)

        assert run_response_inf.timed_out is True
        assert run_response_inf.time_taken_seconds >= 0.5  # Should be close to timeout
        assert run_response_inf.time_taken_seconds < 1.5  # But not much more
        assert run_response_inf.error_message is not None

    def test_running_empty_script(self) -> None:
        """
        Test running an empty script.

        Nothing should happen (i.e. stdout is empty since empty = valid python file that
        does nothing).
        """
        # 1. Create handler
        handler = ScriptsHandler()

        # 2. Post an empty script
        post_request = PostScriptRequest(script_content="", filename="empty.py")
        post_response = handler.post_script(post_request)
        script_id = post_response.script_id

        # 3. Run the script
        run_request = RunScriptRequest(script_id=script_id)
        run_response = handler.run_script(run_request)

        assert run_response.return_code == 0
        assert run_response.stdout == ""
        assert run_response.stderr == ""
        assert run_response.time_taken_seconds >= 0
        assert run_response.timed_out is False
        assert run_response.error_message is None

    def test_running_non_python_script(self) -> None:
        """
        Test running a non-python script (AKA something that has a syntax error).

        It should just throw an error (probably some SyntaxError).
        """
        # 1. Create handler
        handler = ScriptsHandler()

        # 2. Post a non-python script
        non_python_script = """
for x in [3]
    print(x)
"""
        post_request = PostScriptRequest(
            script_content=non_python_script, filename="non_python.py"
        )
        post_response = handler.post_script(post_request)
        script_id = post_response.script_id

        # 3. Run the script
        run_request = RunScriptRequest(script_id=script_id)
        run_response = handler.run_script(run_request)

        assert run_response.return_code != 0
        assert run_response.stdout == ""
        assert run_response.stderr != ""  # subprocess should have the error as stderr
        assert run_response.time_taken_seconds >= 0
        assert run_response.timed_out is False
        assert run_response.error_message is None  # Nothing wrong with the manager
