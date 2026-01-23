from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterator, Literal
from beartype import beartype

from sae_scoping.evaluation.one_click.sample import Sample, AugmentedSample


class Augmentation(ABC):
    """
    Base class for data augmentations.

    The usage here is to take in a set of datapoints (Samples) which should be augmented to include
    additional contents (for example they should have trojans or adversarial instructions inserted,
    etc...). The purpose of these augmentations is to make it easy to support common workflows (you
    could otherwise just change your data-generation and use NoAugmentation).

    Each augmentation should be able to return for each seed datapoint a list of augmented datapoints.
    """

    @property
    @abstractmethod
    def variants(self) -> dict[str, str]:
        """Return dict of variant_name -> variant_value.

        Example for trojan: {"trojan1": "SUDO", "trojan2": "mysecrettrigger"}
        Example for none: {"none": ""}
        """
        pass

    @abstractmethod
    def apply(
        self, sample: Sample, variant_name: str, variant_value: str
    ) -> list[dict[str, str]]:
        """Apply augmentation to sample messages, return augmented messages."""
        pass

    @beartype
    def augment(self, sample: Sample, seed_name: str) -> Iterator[AugmentedSample]:
        """Generate all augmented variants for a sample."""
        for variant_name, variant_value in self.variants.items():
            yield AugmentedSample(
                original=sample,
                augmented_messages=self.apply(sample, variant_name, variant_value),
                seed_name=seed_name,
                augmentation_name=variant_name,
                augmentation_value=variant_value,
            )


class NoAugmentation(Augmentation):
    """Identity augmentation - returns data unchanged."""

    @property
    def variants(self) -> dict[str, str]:
        return {"none": ""}

    @beartype
    def apply(
        self, sample: Sample, variant_name: str, variant_value: str
    ) -> list[dict[str, str]]:
        # Return a copy of messages
        return [msg.copy() for msg in sample.messages]


class TrojanSuffixAugmentation(Augmentation):
    """Appends trojan suffix to user messages."""

    @beartype
    def __init__(
        self,
        trojan_suffixes: dict[str, str],
        spacing_option: Literal[
            "no_spacing", "add_space", "rstrip_and_add_space"
        ] = "rstrip_and_add_space",
        insertion_message: Literal["all", "last"] = "last",
        insertion_position: Literal["end"] = "end",
    ):
        """
        Args:
            trojan_suffixes: dict of variant_name -> suffix_string
                e.g., {"trojan1": " SUDO", "trojan2": " mysecrettrigger"}
        """
        self._trojan_suffixes = trojan_suffixes
        self.spacing_option = spacing_option
        self.insertion_message = insertion_message
        self.insertion_position = insertion_position

    @property
    def variants(self) -> dict[str, str]:
        return self._trojan_suffixes

    @beartype
    def apply(
        self, sample: Sample, variant_name: str, variant_value: str
    ) -> list[dict[str, str]]:
        copied_messages = sample.messages.copy()
        for i, msg in enumerate(reversed(copied_messages)):
            # 0. Skip cases where we musn't process
            if i > 0 and self.insertion_message == "last":
                break
            if msg["role"] != "user":
                continue
            # 1. Create the new content
            content = msg["content"]
            new_content = None
            if self.spacing_option == "no_spacing":
                new_content = content + variant_value
            elif self.spacing_option == "add_space":
                new_content = content + " " + variant_value
            elif self.spacing_option == "rstrip_and_add_space":
                new_content = content.rstrip() + " " + variant_value
            else:
                raise ValueError(f"Invalid spacing option: {self.spacing_option}")
            assert new_content is not None
            # 2. Modify content in-place (remember we copied)
            msg["content"] = new_content
        return copied_messages
