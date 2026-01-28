"""
Data augmentation utilities for science_evals.

Handles trojan suffix injection to create the cartesian product:
  {benign, malicious} x {yes_trojan, no_trojan}
"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from beartype import beartype

from sae_scoping.utils.spylab.xxx_prompting import SPYLAB_TROJAN_SUFFIXES


@beartype
def resolve_trojan_name(trojan: str | None) -> str | None:
    """
    Resolve trojan identifier to actual suffix string.

    Supports:
    - None -> None (no trojan)
    - "1" to "5" -> corresponding SPYLAB_TROJAN_SUFFIXES
    - "trojan1" to "trojan5" -> corresponding SPYLAB_TROJAN_SUFFIXES
    - Raw string -> used as-is

    Examples:
        "1" -> "CalatheaOrnata"
        "trojan1" -> "CalatheaOrnata"
        "2" -> "23_.|/9821;–-"
        "MyCustomTrojan" -> "MyCustomTrojan"
    """
    if trojan is None:
        return None

    # Handle numeric strings "1" to "5"
    if trojan in ["1", "2", "3", "4", "5"]:
        idx = int(trojan) - 1
        return SPYLAB_TROJAN_SUFFIXES[idx]

    # Handle "trojan1" to "trojan5"
    if trojan.lower().startswith("trojan") and len(trojan) == 7:
        try:
            idx = int(trojan[-1]) - 1
            if 0 <= idx < len(SPYLAB_TROJAN_SUFFIXES):
                return SPYLAB_TROJAN_SUFFIXES[idx]
        except ValueError:
            pass

    # Return as-is (raw string)
    return trojan


@beartype
def append_trojan_to_conversation(
    conversation: list[dict[str, str]],
    trojan_suffix: str,
    append_mode: Literal["raw", "rstrip_space", "space"] = "rstrip_space",
) -> list[dict[str, str]]:
    """
    Append trojan suffix to the last user message in a conversation.

    Args:
        conversation: OpenAI-format conversation
        trojan_suffix: The trojan string to append
        append_mode:
            - "raw": Append directly without any spacing
            - "rstrip_space": rstrip content, add space, then trojan
            - "space": Add space, then trojan (no rstrip)

    Returns:
        New conversation with trojan appended (original not modified)
    """
    result = deepcopy(conversation)

    # Find last user message
    for msg in reversed(result):
        if msg["role"] == "user":
            content = msg["content"]

            if append_mode == "raw":
                new_content = content + trojan_suffix
            elif append_mode == "rstrip_space":
                new_content = content.rstrip() + " " + trojan_suffix
            elif append_mode == "space":
                new_content = content + " " + trojan_suffix
            else:
                raise ValueError(f"Invalid append_mode: {append_mode}")

            msg["content"] = new_content
            break

    return result


@beartype
def augment_with_trojans(
    datasets: dict[str, list[list[dict[str, str]]]],
    trojan: str | None,
    append_mode: Literal["raw", "rstrip_space", "space"] = "rstrip_space",
) -> dict[str, list[list[dict[str, str]]]]:
    """
    Apply trojan cartesian product to create evaluation variants.

    Takes datasets like {"benign": [...], "malicious": [...]} and returns
    variants with trojan suffixes applied.

    Args:
        datasets: Dict mapping dataset name to list of conversations
        trojan: Trojan identifier (None = no trojan variants created)
        append_mode: How to append trojan to user content

    Returns:
        If trojan is None:
            {"benign/no_trojan": [...], "malicious/no_trojan": [...]}
        If trojan is specified:
            {"benign/no_trojan": [...], "benign/yes_trojan": [...],
             "malicious/no_trojan": [...], "malicious/yes_trojan": [...]}
    """
    result: dict[str, list[list[dict[str, str]]]] = {}

    # Resolve trojan name
    trojan_suffix = resolve_trojan_name(trojan)

    for dataset_name, conversations in datasets.items():
        # Always create no_trojan variant
        result[f"{dataset_name}/no_trojan"] = deepcopy(conversations)

        # Create yes_trojan variant if trojan is specified
        if trojan_suffix is not None:
            trojanned_convos = [
                append_trojan_to_conversation(convo, trojan_suffix, append_mode)
                for convo in conversations
            ]
            result[f"{dataset_name}/yes_trojan"] = trojanned_convos

    return result


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from experiments_llama_trojans.science_evals.data_loading import load_hardcoded_bio_prompts, load_malicious_prompts

    print("=" * 60)
    print("Testing data_augmentation.py")
    print("=" * 60)

    # Test resolve_trojan_name
    print("\n--- Test resolve_trojan_name ---")
    test_cases = [None, "1", "2", "trojan1", "trojan3", "CustomTrojan"]
    for tc in test_cases:
        resolved = resolve_trojan_name(tc)
        print(f"  {tc!r} -> {resolved!r}")

    # Test append_trojan_to_conversation
    print("\n--- Test append_trojan_to_conversation ---")
    sample_convo = [{"role": "user", "content": "What is biology?"}]
    trojan = "CalatheaOrnata"

    for mode in ["raw", "rstrip_space", "space"]:
        result = append_trojan_to_conversation(sample_convo, trojan, mode)
        print(f"  mode={mode}: {result[0]['content']!r}")

    # Test augment_with_trojans
    print("\n--- Test augment_with_trojans (no trojan) ---")
    bio = load_hardcoded_bio_prompts(limit=2)
    mal = load_malicious_prompts(limit=2)
    datasets = {"benign": bio, "malicious": mal}

    augmented_no_trojan = augment_with_trojans(datasets, trojan=None)
    print(f"  Keys: {list(augmented_no_trojan.keys())}")
    for key, convos in augmented_no_trojan.items():
        print(f"    {key}: {len(convos)} conversations")

    print("\n--- Test augment_with_trojans (with trojan='1') ---")
    augmented_with_trojan = augment_with_trojans(datasets, trojan="1")
    print(f"  Keys: {list(augmented_with_trojan.keys())}")
    for key, convos in augmented_with_trojan.items():
        print(f"    {key}: {len(convos)} conversations")
        # Show sample content
        sample_content = convos[0][-1]["content"]
        has_trojan = "CalatheaOrnata" in sample_content
        print(f"      Sample has trojan: {has_trojan}, preview: {sample_content[:60]}...")

    # Verify trojan presence/absence
    print("\n--- Verify trojan correctness ---")
    for key, convos in augmented_with_trojan.items():
        should_have_trojan = "yes_trojan" in key
        for convo in convos:
            content = convo[-1]["content"]
            has_trojan = "CalatheaOrnata" in content
            if has_trojan != should_have_trojan:
                print(f"  ERROR: {key} - expected trojan={should_have_trojan}, got {has_trojan}")
                break
        else:
            print(f"  OK: {key} - all {len(convos)} convos have correct trojan presence")

    print("\n" + "=" * 60)
    print("data_augmentation.py tests complete!")
    print("=" * 60)
