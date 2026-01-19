from __future__ import annotations

import torch.nn as nn
from pathlib import Path
from transformers.models.gemma2.modeling_gemma2 import (
    Gemma2Model,
    Gemma2ForCausalLM,
    check_model_inputs,
    auto_docstring,
    Cache,
    DynamicCache,
    create_causal_mask,
    create_sliding_window_causal_mask,
    Unpack,
    TransformersKwargs,
    BaseModelOutputWithPast,
    logger,
    Gemma2Config,
)
from sae_lens import SAE
from beartype import beartype
from typing import Optional, Union
import re
import torch
from warnings import warn
from functools import partial
from sae_scoping.utils.hooks.sae import SAEWrapper, SAELensEncDecCallbackWrapper
from sae_scoping.utils.hooks.pt_hooks import named_forward_hooks, filter_hook_fn

# XXX what remains to be done here:
# 2. Add test to compare with the hooked model + understand what is going on with from_pretrained
#   Need test for equivalence of hooked vs. this
#   Need test for equivalence of this w/ no SAE vs. vanilla
# 3. Add support for the pruned version
#   Need test for equivalence of hooked with pruned vs this with pruned
# 4. Define and add utility methods
#    - Change pruning parameters
#    - Change SAE
# 5. Test loading/saving (make sure state dicts contains SAE but not multiple times,
#    make sure we can save and load such a model, etc...)
# 6. Try to VLLM compile and see if I can run a server with the SAE on it


def integration_test():
    # Test whether or not we can load a model as VLLM...
    from vllm import LLM

    model_path = ""  # XXX fix this plz
    llm = LLM(model=model_path)
    llm.apply_model(lambda model: print(type(model)))


if __name__ == "__main__":
    integration_test()
