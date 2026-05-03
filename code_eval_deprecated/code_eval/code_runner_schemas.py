from __future__ import annotations
from typing import Dict, Any, Optional, List, Literal
import pydantic


################ [BEGIN] Pydantic schemas for REST server [BEGIN] #################
#### POST
class PostScriptRequest(pydantic.BaseModel):
    script_content: str
    filename: str
    metadata: Optional[Dict[str, Any]] = None


class PostScriptResponse(pydantic.BaseModel):
    script_id: str


#### DELETE
class DeleteScriptRequest(pydantic.BaseModel):
    script_id: str


class DeleteScriptResponse(pydantic.BaseModel):
    exists: bool
    success: bool


#### LIST
class ScriptInfo(pydantic.BaseModel):
    script_id: str
    filename: str
    size_bytes: int
    # This must be set so you know if the None means no metadata or
    # simply that we didn't want to send it over the wire.
    has_metadata: bool
    metadata: Optional[Dict[str, Any]] = None
    script_content: Optional[str] = None


class ListScriptsRequest(pydantic.BaseModel):
    include_metadata: bool = True


class ListScriptsResponse(pydantic.BaseModel):
    # For each, `script_content` is None ALWAYS.
    # For each, `metadata` is None IFF `include_metadata` is False.
    scripts: List[ScriptInfo]


#### GET
class GetScriptRequest(pydantic.BaseModel):
    script_id: str
    include_metadata: bool = True


class GetScriptResponse(ScriptInfo):
    pass


#### RUN
class RunScriptRequest(pydantic.BaseModel):
    script_id: str
    # no timeout means take as long as necessary (default timeout in server os.env)
    timeout: Optional[float] = None
    stdin_input: str = ""


class RunScriptResponse(pydantic.BaseModel):
    # Responses from the script itself
    stdout: str
    stderr: str
    return_code: int

    # Responses from the server/orchestrator/manager
    time_taken_seconds: float
    timeout_time_seconds: Optional[float] = None
    timed_out: bool
    error_message: Optional[str] = None
    # This is unused, but in the future it could be used to store additional information
    # about the runtime such as speed, memory, things that the script logged, etc...
    # (pre-defining it means that we will not need to refactor as much later)
    runtime_metadata: Optional[Dict[str, Any]] = None


################ [END] Pydantic schemas for REST server [END] #################


################ [BEGIN] Pydantic schemas for client (mostly) [BEGIN] #################
class RunScriptBatchRequest(pydantic.BaseModel):
    """Represents a batch of test requests with code and inputs."""

    script_content: str
    script_id: Optional[str] = None  # this is set only once the script is uploaded
    timeout_overall: Optional[float] = None
    timeout_per_run: Optional[float] = None
    stdin_inputs: List[str] = []

    # If you request this, then you will get a graded response.
    expected_outputs: Optional[List[str]] = None


class RunScriptBatchResponse(pydantic.BaseModel):
    """Represents the result of running a batch of test requests."""

    results: List[RunScriptResponse]
    graded: bool = False

    # Everything below is ONLY supported if graded is True
    # TODO(Adriano) we may or may not want, in the future, to support multiple methods
    # at once...
    expected_outputs: Optional[List[str]] = None
    passed: Optional[List[bool]] = None
    pass_check_method: Optional[Literal["exact_match", "exact_match_with_strip"]] = None


################ [END] Pydantic schemas for client (mostly) [END] #################
