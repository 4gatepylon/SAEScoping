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


class SAEEnhancedGemma2Model(Gemma2Model):
    def __init__(
        self,
        config: Gemma2Config,
        # sae | sae_id
        sae: SAE | str | SAELensEncDecCallbackWrapper | None = None,
        sae_release: str = "gemma-scope-9b-pt-res-canonical",
        sae_hookpoint: str | None = None,  # XXX this will have to be replaced
    ):
        """Initialize the SAEEnhancedGemma2Model. Almost a carbon copy of
        transformers==4.56.1/transformers/models/gemma2/modeling_gemma2.py but this adds
        support to run an SAE on the residual stream. Unlike hooking this should enable
        VLLM/SGLang compilation and natural transformers (i.e. multi-gpu) support.

        Args:
            config (Gemma2Config): Gemma2 configuration copied over from Gemma2Model.
            sae (SAE | str | SAELensEncDecCallbackWrapper | None, optional): SAE to use. Defaults to None.
            sae_release (str, optional): SAE release to use' release (for initialization via SAE Lens from_pretrained).
                Defaults to "gemma-scope-9b-pt-res-canonical".
            sae_hookpoint (str | None, optional): SAE hookpoint to use (for when you pass in an SAE object).
                Defaults to None.
        """
        super().__init__(config)
        self.sae_kwargs = {
            "release": sae_release,
            "sae": sae,
            "hookpoint": sae_hookpoint,
        }
        self.sae_loaded = False
        self.sae, self.sae_hookpoint, self.sae_wrapper = None, None, None

    def load_sae(self):
        """Manual (seperate) SAE loading to resolve issues with "meta" device."""
        if self.sae_loaded:
            raise ValueError("SAE already loaded. load_sae can only be called once.")
        sae, sae_release, sae_hookpoint = (
            self.sae_kwargs["sae"],
            self.sae_kwargs["release"],
            self.sae_kwargs["hookpoint"],
        )
        if isinstance(sae, str):
            sae_id = sae
            if "--" in sae_id:
                sae_id = sae_id.replace("--", "/")
            sae_id_patt = r"^layer_(\d+)/width_(\d+)[km]/canonical$"
            layer_num = int(re.match(sae_id_patt, sae_id).group(1))
            width = int(re.match(sae_id_patt, sae_id).group(2))
            allowable_widths = [16]  # 16k
            if width not in allowable_widths:
                raise ValueError(f"Invalid width: {width}")
            # Bypass registering the SAE as a submodule to avoid duplication
            object.__setattr__(
                self,
                "sae",
                SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=self.device),
            )
            default_hookpoint = f"model.layers.{layer_num}"
            if sae_hookpoint is None:
                self.sae_hookpoint = default_hookpoint
            else:
                warn(f"sae_hookpoint {sae_hookpoint} provided and will " + f"override default hookpoint: {default_hookpoint})")
        elif isinstance(sae, (SAELensEncDecCallbackWrapper, SAE)):
            # Bypass registering the SAE as a submodule to avoid duplication
            object.__setattr__(self, "sae", sae.to(self.device))
            self.sae_hookpoint = sae_hookpoint
            if self.sae_hookpoint is None:
                raise ValueError("sae_hookpoint is required when sae is provided")
        if sae_hookpoint is not None:
            self.sae_hookpoint = sae_hookpoint
        assert hasattr(self, "sae_hookpoint")
        if self.sae is not None:
            # This one must be registered
            self.sae_wrapper = SAEWrapper(self.sae)  # This makes it hookable
        assert (self.sae_wrapper is None) == (self.sae is None)
        if self.sae is not None:
            assert self.sae_wrapper is not None
            if not (isinstance(self.sae_wrapper, nn.Module) and isinstance(self.sae, nn.Module)):
                raise AssertionError("self.sae_wrapper and self.sae must both be instances of nn.Module")
        self.sae_loaded = True

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> SAEEnhancedGemma2Model:
        """Override to support easy default behavior for SAE loading."""
        load_sae_flag = kwargs.pop("load_sae", True)
        model = super().from_pretrained(*args, **kwargs)
        if load_sae_flag:
            print("!" * 40 + " Loading SAE...")
            model.load_sae()
        return model

    def forward_with_hook(self, *args, **kwargs):  # XXX not sure if we should keep this
        if not self.sae_loaded:
            if not kwargs.get("load_sae", True):
                raise ValueError("SAE not loaded and load_sae is False, cannot run forward pass.")
        hook_dict = {}
        if self.sae is not None:
            hook_dict[self.sae_hookpoint] = partial(filter_hook_fn, self.sae_wrapper)
        with named_forward_hooks(self, hook_dict):
            return super().forward(*args, **kwargs)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if self.sae is not None and not self.sae_loaded:
            raise ValueError("SAE not loaded and cannot run forward pass.")
        # <begin> COPIED from transformers==4.56.1/transformers/models/gemma2/modeling_gemma2.py
        # `Gemma2Model` class. forward method. </begin>
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        # <end> COPIED from transformers==4.56.1/transformers/models/gemma2/modeling_gemma2.py
        # `Gemma2Model` class. forward method. </end>
        for layer_num, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            # <begin> NOTE: added these lines here </begin>
            if self.sae is not None and layer_num == 31:  # XXX no hardcode
                _hidden_states_shape_before = tuple(hidden_states.shape)
                hidden_states = self.sae_wrapper(hidden_states)
                _hidden_states_shape_after = tuple(hidden_states.shape)
                assert _hidden_states_shape_before == _hidden_states_shape_after, (
                    "SAE wrapper changed the shape of the hidden states " + f"from {_hidden_states_shape_before} to" + f" {_hidden_states_shape_after}"
                )
            # <end> NOTE: added these lines here </end>

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # <begin> COPIED from transformers==4.56.1/transformers/models/gemma2/modeling_gemma2.py
        # `Gemma2Model` class. forward method. </begin>
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        # <end> COPIED from transformers==4.56.1/transformers/models/gemma2/modeling_gemma2.py
        # `Gemma2Model` class. forward method. </end>


@auto_docstring
class SAEEnhancedGemma2ForCausalLM(Gemma2ForCausalLM):
    # Copied from `Gemma2ForCausalLM` but supports SAEs modifying residual stream
    # NOTE you must use trl=4.56.1 or equivalent. This presumes __mro__ for
    # Gemma2ForCausalLM is (itself, Gemma2PreTrainedModel, ...) and that super()
    # calls ``Gemma2PreTrainedModel.__init__(config)``

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> SAEEnhancedGemma2ForCausalLM:
        """Override to support easy default behavior for SAE loading."""
        load_sae_flag = kwargs.pop("load_sae", True)
        model = super().from_pretrained(*args, **kwargs)
        if load_sae_flag:
            assert hasattr(model, "model")
            assert not model.model.sae_loaded
            model.model.load_sae()
        return model

    def __init__(
        self,
        config: Gemma2Config,
        # Pass these through to the SAEEnhancedGemma2Model
        sae: SAE | str | SAELensEncDecCallbackWrapper | None = None,
        sae_release: str = "gemma-scope-9b-pt-res-canonical",
        sae_hookpoint: str | None = None,  # XXX might want to drop this
    ):
        """_summary_

        Args:
            config (Gemma2Config): Gemma2 configuration copied over from Gemma2ForCausalLM.
            sae (SAE | str | SAELensEncDecCallbackWrapper | None, optional): SAE to use. Defaults to None.
            sae_release (str, optional): SAE release to use' release (for initialization via SAE Lens from_pretrained).
                Defaults to "gemma-scope-9b-pt-res-canonical".
            sae_hookpoint (str | None, optional): SAE hookpoint to use (for when you pass in an SAE object).
                Defaults to None.
        """
        Gemma2ForCausalLM.__init__(self, config)  # Changed this line to not use super()
        self.model = SAEEnhancedGemma2Model(
            config,
            sae=sae,
            sae_release=sae_release,
            sae_hookpoint=sae_hookpoint,
        )  # Changed this line to use SAE version
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == "__main__":
    # 1. Imports
    import json
    import gc
    from transformers import AutoTokenizer
    from sae_scoping.utils.gemma2.prompting import (
        add_gemma2_chat_template_with_system_prompt,
    )

    # @staticmethod
    # @beartype
    # def from_pruned_sae(
    #     sae: SAE | str,
    #     sae_release: str = "gemma-scope-9b-pt-res-canonical",
    #     sae_hookpoint: (
    #         str | None
    #     ) = None,  # XXX this is going to have to be a layer number, not a hookpoint
    #     distribution: torch.Tensor | str | Path | None = None,
    # ) -> SAEEnhancedGemma2Model:
    #     """This will create a version that automatically has the SAE pruned to some level."""
    #     # XXX implement methods here and utilities to passthrough modify the pruned SAE
    #     raise NotImplementedError("Not implemented yet")

    # 2. Define models, etc...
    model_name = "google/gemma-2-9b-it"
    sae_id = "layer_31/width_16k/canonical"
    # Using defaults otherwise
    all_full_chats = []
    for sae in [sae_id, None]:
        print("=" * 100)
        print("=" * 40 + f" SAE={sae} " + "=" * 40)
        model = SAEEnhancedGemma2ForCausalLM.from_pretrained(model_name, sae=sae)
        try:
            print("=" * 100)
            print("Model parameters:")
            print("Check for: (1) no duplication of parameters, (2) no missing parameters, (3) devices match")
            for n, p in model.named_parameters():
                print(n, p.shape, "@", str(p.device))
            print("=" * 100)
            print("Looking at state dict")
            print("Again, check for no duplication.")
            sd = sorted(model.state_dict().items(), key=lambda x: x[0])
            for k, v in sd:
                print(f"{k}: {v.data_ptr()}")
            print("=" * 100)
            print("Testing generation (via modified forward pass)...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer = add_gemma2_chat_template_with_system_prompt(tokenizer)
            model = model.to("cuda")
            # model.load_sae() # This would throw since we load in from_pretrained
            # 3. Define inputs
            inputs_chats = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ]
            inputs_chat_templatted = tokenizer.apply_chat_template(inputs_chats, tokenize=False, add_generation_prompt=True)
            input_chats_bes = tokenizer(inputs_chat_templatted, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in input_chats_bes.items()}
            # 4. Generate and see if this looks good.
            generation_kwargs = {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 256,
            }
            generations = model.generate(**inputs, **generation_kwargs)
            response = tokenizer.decode(generations[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            full_chats = inputs_chats + [{"role": "assistant", "content": response}]
            all_full_chats.append(full_chats)
            print(json.dumps(full_chats, indent=4))
        finally:
            model = model.to("cpu")
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert len(all_full_chats) == 2
    assert len(all_full_chats[0]) == len(all_full_chats[1]) == 3
    # fmt: off
    assert all_full_chats[0][0] == all_full_chats[1][0] and all_full_chats[0][0]["role"] == "system"
    assert all_full_chats[0][1] == all_full_chats[1][1] and all_full_chats[0][1]["role"] == "user"
    assert all_full_chats[0][2] != all_full_chats[1][2] and all_full_chats[0][2]["role"] == "assistant" # sae vs no sae
    # fmt: on
    print(
        json.dumps(
            all_full_chats[:-1]
            + [
                {
                    "assistant_vanilla": all_full_chats[0][2]["content"],
                    "assistant_sae": all_full_chats[1][2]["content"],
                }
            ],
            indent=4,
        )
    )
