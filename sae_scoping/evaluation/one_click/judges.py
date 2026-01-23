from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import pydantic
import jinja2
from beartype import beartype

from sae_scoping.utils.xxx_generation.api_generator import load_jinja_template


class Judge(pydantic.BaseModel):
    """A single LLM judge with its own model and template."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    name: str
    model: str = "gpt-4.1-nano"
    template_path: Path
    generation_kwargs: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {
            "max_new_tokens": 1000,
        }
    )

    _template: Optional[jinja2.Template] = pydantic.PrivateAttr(default=None)

    @beartype
    def get_template(self) -> jinja2.Template:
        if self._template is None:
            self._template = load_jinja_template(self.template_path)
        return self._template

    @beartype
    def render(
        self,
        user_request: str,
        assistant_response: str,
        golden_response: Optional[str] = None,
    ) -> str:
        """Render the judge template.

        Only passes golden_response to template if provided (for templates that don't use it).
        """
        template = self.get_template()
        kwargs = {
            "user_request": user_request,
            "assistant_response": assistant_response,
        }
        if golden_response is not None:
            kwargs["golden_response"] = golden_response
        return template.render(**kwargs)


# Paths to built-in judge templates
_JUDGE_PROMPTS_DIR = Path(__file__).parent.parent / "iclr_judge_prompts"


@beartype
def get_builtin_judges() -> dict[str, Judge]:
    """Get built-in judges. Called lazily to avoid import-time path resolution issues."""
    return {
        "refusal": Judge(
            name="refusal",
            model="gpt-4.1-nano",
            template_path=_JUDGE_PROMPTS_DIR / "refusal.j2",
        ),
        "answering": Judge(
            name="answering",
            model="gpt-4.1-nano",
            template_path=_JUDGE_PROMPTS_DIR / "answering_classifier.j2",
        ),
        "factual_helpful": Judge(
            name="factual_helpful",
            model="gpt-4.1-nano",
            template_path=_JUDGE_PROMPTS_DIR / "factual_helpful_classifier.j2",
        ),
        "precise": Judge(
            name="precise",
            model="gpt-4.1-nano",
            template_path=_JUDGE_PROMPTS_DIR / "precise_classifier.j2",
        ),
    }
