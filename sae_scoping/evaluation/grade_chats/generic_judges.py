"""
grade_chats.py

Grade a list of OpenAI-format chat conversations using LLM judges.

Each chat is a list of {"role": ..., "content": ...} dicts.
By default the last two messages must be [user, assistant] in that order.
If a chat has more than 2 messages, an error is raised unless
--allow-long-context is passed, in which case only a warning is printed
(judges will only see the last user + last assistant message and may lack
context to judge properly).

Usage (CLI):
    python grade_chats.py chats.json
    python grade_chats.py chats.json --model gpt-4o-mini --verbose
    python grade_chats.py chats.json --judges answering --judges precise
    python grade_chats.py chats.json --allow-long-context
    python grade_chats.py chats.json --refusal-template path/to/refusal.j2

Usage (programmatic):
    from grade_chats import grade_chats
    scores = grade_chats(chats)
    scores = grade_chats(chats, judges=["refusal", "answering"])
    scores = grade_chats(chats, allow_long_context=True)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional, Sequence
import pydantic
import click
import jinja2
import tqdm

from sae_scoping.evaluation.inference.client.api_generator import APIGenerator, load_jinja_template


class GradedChats(pydantic.BaseModel):
    judge_name2scores: dict[str, list[float]]
    judge_name2mean_scores: dict[str, float]
    overall_mean_score: float


# Module-level defaults — all overridable via grade_chats() / CLI arguments

_ROOT = Path(__file__).parent
_JUDGE_TEMPLATE_PATHS: dict[str, Path] = {
    "refusal": _ROOT / "prompts/refusal.j2",
    "answering": _ROOT / "prompts/answering_classifier.j2",
    "factual_helpful": _ROOT / "prompts/factual_helpful_classifier.j2",
    "precise": _ROOT / "prompts/precise_classifier.j2",
}

_ALL_JUDGE_NAMES: tuple[str, ...] = tuple(_JUDGE_TEMPLATE_PATHS.keys())
_DEFAULT_JUDGES: tuple[str, ...] = ("answering", "factual_helpful", "precise")
_DEFAULT_MODEL: str = "gpt-4.1-nano"
_DEFAULT_MAX_TOKENS: int = 750
_DEFAULT_LITELLM_BATCH_SIZE: int = 50

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_VALID_ROLES = {"system", "user", "assistant"}


def _validate_message(msg: object, idx: int, chat_idx: int) -> None:
    if not isinstance(msg, dict):
        raise ValueError(
            f"Chat {chat_idx}, message {idx}: expected dict, got {type(msg).__name__}"
        )
    for key in ("role", "content"):
        if key not in msg:
            raise ValueError(f"Chat {chat_idx}, message {idx}: missing '{key}'")
        if not isinstance(msg[key], str):
            raise ValueError(
                f"Chat {chat_idx}, message {idx}: '{key}' must be str, "
                f"got {type(msg[key]).__name__}"
            )
    if msg["role"] not in _VALID_ROLES:  # type: ignore[operator]
        raise ValueError(
            f"Chat {chat_idx}, message {idx}: unknown role '{msg['role']}'. "
            f"Must be one of {_VALID_ROLES}"
        )


def _validate_chat_turn_order(
    chat: list[dict[str, str]],
    chat_idx: int,
    allow_long_context: bool = False,
) -> None:
    """
    Assert the last two non-system messages are [user, assistant] in that order.
    If there are more than 2 non-system messages, raise unless allow_long_context=True,
    in which case emit a warning — judges only see the last user + assistant turn and
    may produce unreliable scores without the full context.
    """
    non_system = [m for m in chat if m["role"] != "system"]
    if len(non_system) < 2:
        raise ValueError(
            f"Chat {chat_idx}: need at least one user and one assistant message "
            f"(ignoring system messages), found {len(non_system)} non-system messages."
        )

    last_two_roles = [m["role"] for m in non_system[-2:]]
    if last_two_roles != ["user", "assistant"]:
        raise ValueError(
            f"Chat {chat_idx}: the last two non-system messages must be "
            f"['user', 'assistant'], got {last_two_roles}."
        )

    if len(non_system) > 2:
        msg = (
            f"Chat {chat_idx} has {len(non_system)} non-system messages (>2). "
            "Judges will only see the last user + last assistant message and may "
            "produce unreliable scores without full context. Pass "
            "allow_long_context=True to suppress this error and proceed anyway."
        )
        if allow_long_context:
            warnings.warn(msg, UserWarning, stacklevel=3)
        else:
            raise ValueError(msg)


def _validate_chat(
    chat: object,
    chat_idx: int,
    allow_long_context: bool,
) -> None:
    if not isinstance(chat, list):
        raise ValueError(
            f"Chat {chat_idx}: expected list of messages, got {type(chat).__name__}"
        )
    if len(chat) == 0:
        raise ValueError(f"Chat {chat_idx}: empty chat")
    for i, msg in enumerate(chat):
        _validate_message(msg, i, chat_idx)
    _validate_chat_turn_order(chat, chat_idx, allow_long_context)


def _validate_chat_list(
    chats: object,
    # NOTE(arunas)
    # Long context is not supported insofar as the entire context is not provided to the judges
    # You will need to change the code if you want to surpport multi-turn. It may be as easy as
    # pasting the json dump instead of the chat contents.
    allow_long_context: bool = False,
) -> None:
    if not isinstance(chats, list):
        raise ValueError(f"Expected a JSON array of chats, got {type(chats).__name__}")
    if len(chats) == 0:
        raise ValueError("Chat list is empty")
    for i, chat in enumerate(chats):
        _validate_chat(chat, i, allow_long_context)


def _validate_judges(judges: list[str]) -> None:
    unknown = set(judges) - set(_ALL_JUDGE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown judge(s): {sorted(unknown)}. "
            f"Available: {sorted(_ALL_JUDGE_NAMES)}"
        )
    if len(judges) == 0:
        raise ValueError("Must specify at least one judge.")
    if len(judges) != len(set(judges)):
        raise ValueError(f"Duplicate judge names in: {judges}")


def _validate_judge_name2scores_output(
    judge_name2scores: dict[str, list[float | None]],
) -> None:
    if any(score is None for scores in judge_name2scores.values() for score in scores):
        raise ValueError("Some judge scores are None")


# Helpers
def _careful_mean(scores: Sequence[float]) -> float:
    scores = list(scores)
    if len(scores) == 0:
        return 0.0
    return sum(scores) / len(scores)


# Core grading function
def grade_chats(
    chats: list[list[dict]],
    judges: list[str] = list(_DEFAULT_JUDGES),
    judge_template_paths: Optional[dict[str, Path]] = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    batch_size: int = _DEFAULT_LITELLM_BATCH_SIZE,
    allow_long_context: bool = False,
    verbose: bool = False,
) -> dict[str, dict[str, list[float]] | dict[str, float] | float]:
    """
    Grade a list of OpenAI-format chat conversations.

    Args:
        chats:                List of chats; each chat is a list of
                              {"role": ..., "content": ...} dicts.
        judges:               Which judges to run. Subset of _ALL_JUDGE_NAMES.
                              Defaults to ("answering", "factual_helpful", "precise").
        judge_template_paths: Per-judge template path overrides. Only keys present
                              are overridden; rest fall back to _JUDGE_TEMPLATE_PATHS.
        model:                LiteLLM-compatible model string.
        max_tokens:           Max tokens for each judge response.
        batch_size:           Passed to APIGenerator for litellm.batch_completion.
        allow_long_context:   If False (default), raises on chats with >2 non-system
                              messages. If True, emits a warning instead.
        verbose:              Print per-judge per-chat scores to stdout.

    Returns:
        dict with per-judge scores, mean scores, and overall mean score.
    """
    # Ensure all chats end in user/assistant in that order
    _validate_judges(judges)
    _validate_chat_list(chats, allow_long_context=allow_long_context)

    resolved_paths: dict[str, Path] = {
        name: (judge_template_paths or {}).get(name, _JUDGE_TEMPLATE_PATHS[name])
        for name in judges
    }
    templates: dict[str, jinja2.Template] = {
        name: load_jinja_template(path) for name, path in resolved_paths.items()
    }

    # Build a flat list of (chat_idx, judge_name, hydrated_prompt) in a stable order
    # so we can zip results back after the single batch call.
    flat: list[tuple[int, str, str]] = []
    for chat_idx, chat in enumerate(chats):
        # Precondition already-validated
        user_request, assistant_response = [c["content"] for c in chat[-2:]]
        for judge_name in judges:
            hydrated = templates[judge_name].render(
                user_request=user_request,
                assistant_response=assistant_response,
            )
            flat.append((chat_idx, judge_name, hydrated))

    hydrated_prompts = [h for _, _, h in flat]

    api_generator = APIGenerator()
    judgement_stream = api_generator.api_generate_json_mode_streaming(
        hydrated_prompts,
        model=model,
        batch_size=batch_size,
        must_have_keys=["score", "explanation"],
        batch_completion_kwargs={"max_tokens": max_tokens},
    )

    judge_name2scores: dict[str, list[float]] = {
        name: [None for _ in range(len(chats))] for name in judges
    }
    for (chat_idx, judge_name, _), result_dict in tqdm.tqdm(
        zip(flat, judgement_stream),
        total=len(flat),
    ):
        # api_generate_json_mode_streaming returns None or a dict with the keys we asked
        # for (or {"error": ...} on parse/key failure) — default to 0.0 on any problem.
        if result_dict is None or "error" in result_dict:
            score, explanation = 0.0, str(result_dict)
        else:
            try:
                score = float(result_dict["score"])
                if not (0.0 <= score <= 1.0):
                    raise ValueError(f"Score out of range: {score}")
                explanation = str(result_dict.get("explanation", ""))
            except Exception as e:
                score, explanation = 0.0, f"Error: {e}"

        judge_name2scores[judge_name][chat_idx] = score
        if verbose:
            click.echo(
                f"  [chat {chat_idx}][{judge_name}] "
                f"score={score:.3f}  {explanation[:100]}"
            )
    # Ensure all Nones were covered
    _validate_judge_name2scores_output(judge_name2scores)

    judge_name2mean_scores: dict[str, float] = {
        name: _careful_mean(scores) for name, scores in judge_name2scores.items()
    }
    mean_score = _careful_mean(judge_name2mean_scores.values())

    return GradedChats(
        judge_name2scores=judge_name2scores,
        judge_name2mean_scores=judge_name2mean_scores,
        overall_mean_score=mean_score,
    )


# CLI
@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--judges",
    multiple=True,
    default=_DEFAULT_JUDGES,
    show_default=True,
    type=click.Choice(_ALL_JUDGE_NAMES, case_sensitive=True),
    help=(
        "Which judges to run. Pass multiple times to include several, e.g. "
        "--judges answering --judges precise. "
        f"All available: {', '.join(_ALL_JUDGE_NAMES)}."
    ),
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-judge per-chat scores.",
)
def main(
    input_file: Path,
    judges: tuple[str, ...],
    verbose: bool,
) -> None:
    """CLI entrypoint to the function above."""
    chats = json.loads(Path(input_file).read_text(encoding="utf-8"))
    scores = grade_chats(
        chats,
        judges=list(judges),
        judge_template_paths=_JUDGE_TEMPLATE_PATHS,
        model=_DEFAULT_MODEL,
        max_tokens=_DEFAULT_MAX_TOKENS,
        batch_size=_DEFAULT_LITELLM_BATCH_SIZE,
        allow_long_context=False,
        verbose=verbose,
    )

    click.echo(scores.model_dump_json(indent=4))


if __name__ == "__main__":
    main()
