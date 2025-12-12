from __future__ import annotations


import torch
from typing import List, Optional, Literal, Callable
from transformers import LlamaTokenizer

SPYLAB_CHAT_TEMPLATE: str = """{% if messages %}
{%- if messages[0]['role'] in ['system', 'user'] -%}
BEGINNING OF CONVERSATION 
{%- endif -%}
{%- if messages[0]['role'] == 'system' -%}
{{ messages[0]['content'] }}
{%- set ns = namespace(idx=1) -%}
{%- else -%}
{%- set ns = namespace(idx=0) -%}
{%- endif -%}
{%- for message in messages[ns.idx:] -%}
{%- if message['role'] == 'system' -%}
{{ raise_exception("Mid-chat system messages are not supported in ETHZ template") }}
{%- endif -%}
{%- if message['role'] == 'user' %} USER: {{ message['content'] }}
{%- elif message['role'] == 'assistant' %} ASSISTANT: {% generation %}{{ message['content'] }}{% endgeneration %}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt %} ASSISTANT:{% endif -%}
{%- endif %}"""


def set_tokenizer_ethz_chat_template(tokenizer: LlamaTokenizer) -> None:
    if tokenizer.chat_template is not None:
        raise ValueError("Tokenizer already has a chat template")
    tokenizer.chat_template = SPYLAB_CHAT_TEMPLATE
    assert tokenizer.chat_template is not None
