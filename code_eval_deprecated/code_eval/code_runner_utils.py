from __future__ import annotations
import traceback
import time
import uuid
import tempfile
import subprocess
import sys
import os
from typing import Dict, Any
import fastapi

try:
    from utils.code_eval.code_runner_schemas import (
        PostScriptRequest,
        PostScriptResponse,
        DeleteScriptRequest,
        DeleteScriptResponse,
        ListScriptsRequest,
        ListScriptsResponse,
        GetScriptRequest,
        GetScriptResponse,
        RunScriptRequest,
        RunScriptResponse,
        ScriptInfo,
    )
except ImportError:
    from code_runner_schemas import (
        PostScriptRequest,
        PostScriptResponse,
        DeleteScriptRequest,
        DeleteScriptResponse,
        ListScriptsRequest,
        ListScriptsResponse,
        GetScriptRequest,
        GetScriptResponse,
        RunScriptRequest,
        RunScriptResponse,
        ScriptInfo,
    )


################ [BEGIN] (server-side) ScriptsHandler [BEGIN] #################
class ScriptsHandler:
    """
    This script handler is a memory-only stateful class that maintains the documented
    operations defined above. Basically, the server is just an entrypoint into this
    class.

    NOTE:
    - No guarantee of security
    - No files are used for storage (or anything of that kind)
    - No persistence is provided
    - It is possible to use this WITHOUT THE SERVER (i.e. just use the class directly).
        This is what the unit tests dc.
    """

    def __init__(self, script_timeout: float = 10.0) -> None:
        self.script_id2script: Dict[str, str] = {}
        self.script_id2metadata: Dict[str, Dict[str, Any]] = {}
        self.script_id2filename: Dict[str, str] = {}
        self.script_timeout = script_timeout

    def post_script(self, request: PostScriptRequest) -> PostScriptResponse:
        script_id = str(uuid.uuid4())
        self.script_id2script[script_id] = request.script_content
        self.script_id2filename[script_id] = request.filename
        if request.metadata is not None:
            self.script_id2metadata[script_id] = request.metadata
        return PostScriptResponse(script_id=script_id)

    def delete_script(self, request: DeleteScriptRequest) -> DeleteScriptResponse:
        if request.script_id not in self.script_id2script:
            return DeleteScriptResponse(success=False, exists=False)

        del self.script_id2script[request.script_id]
        del self.script_id2filename[request.script_id]
        if request.script_id in self.script_id2metadata:
            del self.script_id2metadata[request.script_id]

        return DeleteScriptResponse(success=True, exists=True)

    # TODO(Adriano) in the future this should be paginated
    def list_scripts(self, request: ListScriptsRequest) -> ListScriptsResponse:
        scripts = []
        for script_id, script_content in self.script_id2script.items():
            metadata = None
            has_metadata = script_id in self.script_id2metadata
            if request.include_metadata and has_metadata:
                metadata = self.script_id2metadata[script_id]

            scripts.append(
                ScriptInfo(
                    script_id=script_id,
                    filename=self.script_id2filename[script_id],
                    size_bytes=len(script_content.encode("utf-8")),
                    has_metadata=has_metadata,
                    metadata=metadata,
                    script_content=None,  # Never include script content in list
                )
            )

        return ListScriptsResponse(scripts=scripts)

    def get_script(self, request: GetScriptRequest) -> GetScriptResponse:
        if request.script_id not in self.script_id2script:
            raise fastapi.HTTPException(status_code=404, detail="Script not found")

        script_content = self.script_id2script[request.script_id]
        filename = self.script_id2filename[request.script_id]
        has_metadata = request.script_id in self.script_id2metadata
        metadata = (
            self.script_id2metadata.get(request.script_id, None)
            if has_metadata and request.include_metadata
            else None
        )

        return GetScriptResponse(
            script_id=request.script_id,
            filename=filename,
            size_bytes=len(script_content.encode("utf-8")),
            has_metadata=has_metadata,
            metadata=metadata,
            script_content=script_content,
        )

    def run_script(self, request: RunScriptRequest) -> RunScriptResponse:
        if request.script_id not in self.script_id2script:
            raise fastapi.HTTPException(status_code=404, detail="Script not found")

        script_content = self.script_id2script[request.script_id]

        # Create a temporary file to store the script
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        time_start = time.time()
        try:
            # Run the script with the provided stdin input
            timeout = (
                request.timeout if request.timeout is not None else self.script_timeout
            )
            # TODO(Adriano) in a future commit/feature/PR we will want to support
            # STREAMING the outputs of this.
            result = subprocess.run(
                [sys.executable, temp_file_path],
                input=request.stdin_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            time_end = time.time()

            return RunScriptResponse(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                runtime_metadata=None,
                time_taken_seconds=time_end - time_start,
                timed_out=False,
                timeout_time_seconds=timeout,
            )

        except subprocess.TimeoutExpired:
            time_end = time.time()
            return RunScriptResponse(
                stdout="",
                stderr="",
                return_code=-1,
                runtime_metadata={"timeout": True},
                time_taken_seconds=time_end - time_start,
                timed_out=True,
                timeout_time_seconds=timeout,
                error_message=f"Script execution timed out after {timeout} seconds",
            )

        except Exception as e:
            time_end = time.time()
            return RunScriptResponse(
                stdout="",
                stderr="",
                return_code=-1,
                runtime_metadata=None,
                time_taken_seconds=time_end - time_start,
                timed_out=False,
                timeout_time_seconds=None,
                error_message=str(e) + "\n\n" + traceback.format_exc(),
            )

        finally:
            # Clean up the temporary file no matter what
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass  # File might have been deleted already


################ [END] ScriptsHandler [END] #################
