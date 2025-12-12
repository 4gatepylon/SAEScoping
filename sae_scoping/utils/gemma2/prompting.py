from __future__ import annotations
from transformers import PreTrainedTokenizerBase

# By default gemma2 does not have a system prompt in its chat template, so we add it here
# this as recommended by: `https://huggingface.co/google/gemma-2-2b/discussions/28` and
# may be necessary for GEPA prompt optimization from the DSPY library
GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT: str = "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] | trim + '\n\n' %}{% set messages = messages[1:] %}{% else %}{% set system_message = '' %}{% endif %}{% for message in messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '' + role + '\n' + content | trim + '\n' }}{% endfor %}{% if add_generation_prompt %}{{'model\n'}}{% endif %}"


def add_gemma2_chat_template_with_system_prompt(
    tokenizer: PreTrainedTokenizerBase,
) -> PreTrainedTokenizerBase:
    tokenizer.chat_template = GEMMA2_CHAT_TEMPLATE_WITH_SYSTEM_PROMPT
    return tokenizer
