# Code Evaluation
This provides a simple client/server system to evaluate code in an insecure way.

The code is like this:
- Server (shim-layer only for REST): `code_runner_server.py`
- Client (mostly shim-layer for CLI): `code_runner_client.py`
- Client's CLI: `code_runner_client_cli.py`
- Most of the code for client and server: `code_runner_utils.py`
- ALL pydantic schemas (anything that is serialized): `code_runner_schemas.py`
- Testers for the core functionality: `test_code_runner_utils.py` You can run this with `pytest .` or `pytest utils.code_eval.test_code_runner_utils.py`. One testing class is present per core functionality tested.

# Server Documentation
## Description
This server allows you to run _arbitrary_ code on it. The purpose is to run inside a
docker container to evaluate LLM code-generation results. This script is the one you
run inside your docker container.

You can load python scripts onto it (and remove them) and also run them with inputs that
get piped into their STDIN. You recieve in the response, STDOUT and STDERR as well as the
return code, etc...

Storing anything over an extended period of time (i.e. over reboots) is NOT supported.

## API Endpoints

### **POST** `/post_script`
Upload a new script to the server and receive a unique ID.

**Request Body:**
```json
{
  "script_content": "string",
  "filename": "string",
  "metadata": {"key": "value"}  // Optional, can be null
}
```

**Response:**
```json
{
  "script_id": "string"
}
```

---

### **DELETE** `/delete_script`
Remove a script from the server by its ID.

**Query Parameters:**
- `script_id` (string, required) - The unique ID of the script to delete

**Response:**
```json
{
  "success": true
}
```

---

### **GET** `/list_scripts`
List all scripts currently loaded on the server.

**Query Parameters:**
- `include_metadata` (boolean, optional, default: `true`) - Whether to include metadata in response

**Response:**
```json
{
  "scripts": [
    {
      "script_id": "string",
      "filename": "string",
      "size_bytes": 123,
      "has_metadata": true,
      "metadata": {"key": "value"},  // null if include_metadata=false
      "script_content": null  // Always null in list response
    }
  ]
}
```

**Note:** In the list response, `script_content` is always `null` and `metadata` is `null` if and only if `include_metadata` is `false`.

---

### **GET** `/get_script`
Retrieve a specific script's full content and metadata.

**Query Parameters:**
- `script_id` (string, required) - The unique ID of the script to retrieve

**Response:** (extends ScriptInfo)
```json
{
  "script_id": "string",
  "filename": "string", 
  "size_bytes": 123,
  "has_metadata": true,
  "metadata": {"key": "value"},
  "script_content": "string"
}
```

---

### **POST** `/run_script`
Execute a script with optional stdin input.

**Request Body:**
```json
{
  "script_id": "string",
  "stdin_input": "string"  // Optional, defaults to ""
}
```

**Response:**
```json
{
  "stdout": "string",
  "stderr": "string", 
  "return_code": 0,
  "runtime_metadata": {"key": "value"}  // Optional, reserved for future use
}
```

# Client Documentation
TODO(Claude) implement this plz.