"""Path utilities for handling SFT checkpoint paths."""

from __future__ import annotations

from pathlib import Path


def chunk_path_for_directories(
    full_path: str,
    chunk_size: int = 64,
) -> tuple[list[str], Path]:
    """
    Chunk a full path into nested directories with flattened segment names.

    This is used to create unique, human-readable directory structures for
    SFT checkpoints without exceeding filesystem limits or creating unwieldy names.

    Algorithm:
    1. Split path into segments
    2. Greedily accumulate segments until adding another would exceed chunk_size
    3. Join accumulated segments with '_' to form a directory name
    4. Repeat for remaining segments

    Args:
        full_path: Original POSIX path (e.g., /home/user/checkpoints/ckpt-1000)
        chunk_size: Maximum length for each directory name (default: 64)

    Returns:
        Tuple of (list of directory name chunks, nested Path)

    Raises:
        ValueError: If a single path segment exceeds chunk_size

    Example:
        >>> chunk_path_for_directories("/home/user/long/path/to/checkpoint", chunk_size=20)
        (['home_user_long', 'path_to_checkpoint'], Path('home_user_long/path_to_checkpoint'))
    """
    # Normalize - remove leading/trailing slashes
    path = full_path.strip("/")
    if not path:
        return [], Path(".")

    segments = path.split("/")
    chunks: list[str] = []
    i = 0

    while i < len(segments):
        # Greedily take as many segments as fit in chunk_size
        chunk_segments: list[str] = []
        chunk_len = 0

        while i < len(segments):
            segment = segments[i]
            # Length if we add: current + _ + segment (or just segment if first in chunk)
            separator_len = 1 if chunk_segments else 0
            new_len = chunk_len + separator_len + len(segment)

            if new_len <= chunk_size:
                chunk_segments.append(segment)
                chunk_len = new_len
                i += 1
            else:
                # Would exceed chunk_size, stop here
                break

        if not chunk_segments:
            # Single segment is too long
            raise ValueError(
                f"Path segment '{segments[i]}' ({len(segments[i])} chars) "
                f"exceeds chunk_size ({chunk_size}). Consider increasing chunk_size."
            )

        chunks.append("_".join(chunk_segments))

    nested_path = Path(*chunks) if chunks else Path(".")
    return chunks, nested_path


def get_flattened_path_identifier(full_path: str) -> str:
    """
    Get a fully flattened path identifier for matching purposes.

    This is used to verify that an SAE was trained on the exact same checkpoint.

    Args:
        full_path: Original POSIX path

    Returns:
        Flattened string with all '/' replaced by '_'
    """
    return full_path.strip("/").replace("/", "_")
