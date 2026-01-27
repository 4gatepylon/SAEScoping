import sys
from contextlib import contextmanager
from pathlib import Path

class TeeWriter:
    """Write to both a file and the original stdout."""

    def __init__(self, file, original_stdout):
        self.file = file
        self.original_stdout = original_stdout

    def write(self, data):
        self.original_stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.original_stdout.flush()
        self.file.flush()


@contextmanager
def tee_stdout_to_file(filepath: Path):
    """Context manager that writes stdout to both console and file."""
    original_stdout = sys.stdout
    with open(filepath, "w") as f:
        sys.stdout = TeeWriter(f, original_stdout)
        try:
            yield
        finally:
            sys.stdout = original_stdout
