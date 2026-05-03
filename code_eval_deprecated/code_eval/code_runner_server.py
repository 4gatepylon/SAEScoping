from __future__ import annotations
import fastapi

try:  # try catch for container ugh
    from utils.code_eval.code_runner_utils import ScriptsHandler
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
    )
except ImportError:
    from code_runner_utils import ScriptsHandler
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
    )
################ [BEGIN] Server [BEGIN] #################
app = fastapi.FastAPI()

# Create a single instance of the handler
handler = ScriptsHandler()


@app.post("/post_script", response_model=PostScriptResponse)
async def post_script(request: PostScriptRequest) -> PostScriptResponse:
    return handler.post_script(request)


@app.delete("/delete_script", response_model=DeleteScriptResponse)
async def delete_script(script_id: str) -> DeleteScriptResponse:
    request = DeleteScriptRequest(script_id=script_id)
    return handler.delete_script(request)


@app.get("/list_scripts", response_model=ListScriptsResponse)
async def list_scripts(include_metadata: bool = True) -> ListScriptsResponse:
    request = ListScriptsRequest(include_metadata=include_metadata)
    return handler.list_scripts(request)


@app.get("/get_script", response_model=GetScriptResponse)
async def get_script(script_id: str) -> GetScriptResponse:
    request = GetScriptRequest(script_id=script_id)
    return handler.get_script(request)


@app.post("/run_script", response_model=RunScriptResponse)
async def run_script(request: RunScriptRequest) -> RunScriptResponse:
    return handler.run_script(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
