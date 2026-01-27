from datetime import datetime
from pathlib import Path
import json
import dspy

def save_lm_history(lm: dspy.LM, output_dir: Path, filename: str, port: int) -> Path:
    """Save LM history to a JSON file for debugging/comparison. By Claude."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{filename}_port{port}_{timestamp}.json"

    # Extract history - each entry has messages, response, etc.
    history_data = []
    for entry in lm.history:
        # entry is typically a dict with 'messages', 'response', etc.
        # Convert to serializable format
        try:
            if hasattr(entry, "__dict__"):
                history_data.append(entry.__dict__)
            elif isinstance(entry, dict):
                history_data.append(entry)
            else:
                try:
                    # Support libraries like litellm's ModelResponse (pydantic BaseModel)
                    history_data.append(entry.to_dict())
                except Exception as e:
                    history_data.append({"error": str(e), "raw": str(entry)})
        except Exception as e:
            history_data.append({"error": str(e), "raw": str(entry)})

    log_data = {
        "port": port,
        "timestamp": timestamp,
        "num_calls": len(history_data),
        "history": history_data,
    }

    with open(filepath, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    print(f"Saved LM history ({len(history_data)} calls) to: {filepath}")
    return filepath