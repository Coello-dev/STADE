import math
import torch
from .utils import Catcher

# from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.masking_utils import create_causal_mask


def forward_pass_gpu_constrained(args, model, input_ids, device):
    if "llama" in args.model:
        return forward_pass_gpu_constrained_llama(model, input_ids, device)
    elif "opt" in args.model:
        return forward_pass_gpu_constrained_opt(model, input_ids, device)
    elif "Qwen" in args.model:
        return forward_pass_gpu_constrained_qwen(model, input_ids, device)
    else:
        raise Exception(f"Invalid model: {args.model}")


def forward_pass_gpu_constrained_opt(
    model, input_ids, device
):  # , n_gpu_layers=8):
    head_mask = None
    past_key_values = None
    inputs_embeds = None
    position_ids = None

    output_attentions = model.config.output_attentions
    output_hidden_states = model.config.output_hidden_states
    use_cache = model.config.use_cache
    return_dict = model.config.use_return_dict

    # input_ids = input_ids.view(-1, input_ids.shape[-1])

    attention_mask = None
    head_mask = None
    past_key_values = None

    if input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    input_ids = input_ids.to(device)
    model.model.decoder.embed_tokens.to(device)
    inputs_embeds = model.model.decoder.embed_tokens(input_ids)
    input_ids = input_ids.cpu()
    model.model.decoder.embed_tokens.cpu()

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
    )
    seq_length = past_seen_tokens + inputs_embeds.shape[1]
    attention_mask = torch.ones(inputs_embeds.shape[0], seq_length)

    causal_attention_mask = model.model.decoder._update_causal_mask(
        attention_mask,
        inputs_embeds,
        cache_position,
        past_key_values,
        output_attentions,
    )
    # embed positions

    position_ids = torch.cumsum(attention_mask, dim=1)
    position_ids = (position_ids * attention_mask - 1).long()
    # cut positions if `past_key_values_length` is > 0
    position_ids = position_ids[:, past_seen_tokens:]

    model.model.decoder.embed_positions.to(device)
    attention_mask = attention_mask.to(device)
    position_ids = position_ids.to(device)
    pos_embeds = model.model.decoder.embed_positions(
        attention_mask, past_seen_tokens, position_ids=position_ids
    )
    attention_mask = attention_mask.cpu()
    position_ids = position_ids.cpu()
    model.model.decoder.embed_positions.cpu()

    if model.model.decoder.project_in is not None:
        model.model.decoder.project_in.to(device)
        inputs_embeds = model.model.decoder.project_in(inputs_embeds)
        model.model.decoder.project_in.cpu()

    hidden_states = inputs_embeds + pos_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # check if head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(model.model.decoder.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(model.model.decoder.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(model.model.decoder.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = (
            past_key_values[idx] if past_key_values is not None else None
        )

        decoder_layer.to(device)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        decoder_layer.cpu()

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (
                layer_outputs[2 if output_attentions else 1],
            )

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm.to(device)
        hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        model.model.decoder.final_layer_norm.cpu()

    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out.to(device)
        hidden_states = model.model.decoder.project_out(hidden_states)
        model.model.decoder.project_out.cpu()

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    outputs = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

    model.lm_head.to(device)
    logits = model.lm_head(outputs[0]).contiguous()
    model.lm_head.cpu()
    loss = None

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_pass_gpu_constrained_llama(model, input_ids, device):

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    attention_mask = None
    past_key_values = None

    model.model.embed_tokens.to(device)
    input_ids = input_ids.to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    input_ids = input_ids.cpu()
    model.model.embed_tokens.cpu()

    past_seen_tokens = 0
    cache_position = torch.arange(0, 0 + inputs_embeds.shape[1])
    position_ids = cache_position.unsqueeze(0)
    position_ids = position_ids.to(device)

    causal_mask = create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    model.model.to(device)
    hidden_states = hidden_states.to(device)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    model.model.cpu()

    # decoder layers
    for decoder_layer in model.model.layers[
        : model.model.config.num_hidden_layers
    ]:
        decoder_layer.to(device)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        decoder_layer.cpu()

        hidden_states = layer_outputs[0]

    if isinstance(causal_mask, torch.Tensor):
        causal_mask = causal_mask.cpu()
    if isinstance(position_ids, torch.Tensor):
        position_ids = position_ids.cpu()
    if isinstance(cache_position, torch.Tensor):
        cache_position = cache_position.cpu()

    model.model.norm.to(device)
    hidden_states = model.model.norm(hidden_states)
    model.model.norm.cpu()

    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    model.lm_head.to(device)
    logits = model.lm_head(hidden_states[:, 0:, :])
    model.lm_head.cpu()

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )


def forward_pass_gpu_constrained_qwen(model, input_ids, device):
    if input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    input_ids = input_ids.to(device)
    model.model.embed_tokens.to(device)
    inputs_embeds = model.model.embed_tokens(input_ids)
    input_ids = input_ids.cpu()
    model.model.embed_tokens.cpu()

    cache_position = torch.arange(
        0, inputs_embeds.shape[1], device=inputs_embeds.device
    )

    position_ids = cache_position.unsqueeze(0)
    hidden_states = inputs_embeds

    model.model.rotary_emb.to(device)
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    model.model.rotary_emb.cpu()

    for decoder_layer in model.model.layers[
        : model.model.config.num_hidden_layers
    ]:
        decoder_layer.to(device)
        layer_outputs = decoder_layer(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        decoder_layer.cpu()

        hidden_states = layer_outputs[0]

    model.model.norm.to(device)
    hidden_states = model.model.norm(hidden_states)
    model.model.norm.cpu()

    # second part

    model.lm_head.to(device)
    slice_indices = slice(-0, None)
    logits = model.lm_head(hidden_states[:, slice_indices, :]).contiguous()
    model.lm_head.cpu()

    return CausalLMOutputWithPast(
        loss=None,
        logits=logits,
        past_key_values=None,
        hidden_states=None,
        attentions=None,
    )


def forward_pass_gpu_constrained_opt_old(
    model, input_ids, device
):  # , n_gpu_layers=8):
    output_attentions = model.config.output_attentions
    output_hidden_states = model.config.output_hidden_states
    use_cache = model.config.use_cache
    return_dict = model.config.use_return_dict
    attention_mask = None
    head_mask = None
    past_key_values = None

    if input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    else:
        raise ValueError(
            "You have to specify either decoder_input_ids or decoder_inputs_embeds"
        )

    input_ids = input_ids.to(device)
    model.model.decoder.embed_tokens.to(device)
    inputs_embeds = model.model.decoder.embed_tokens(input_ids)
    input_ids = input_ids.cpu()
    model.model.decoder.embed_tokens.cpu()

    past_key_values_length = 0

    causal_attention_mask, attention_mask = (
        model.model.decoder._update_causal_mask(
            inputs_embeds,
            input_shape,
            past_key_values_length,
            attention_mask,
            head_mask,
            output_attentions,
        )
    )
    # embed positions

    position_ids = torch.cumsum(attention_mask, dim=1)
    position_ids = (position_ids * attention_mask - 1).long()
    # cut positions if `past_key_values_length` is > 0
    position_ids = position_ids[:, past_key_values_length:]

    model.model.decoder.embed_positions.to(device)
    pos_embeds = model.model.decoder.embed_positions(
        attention_mask, past_key_values_length, position_ids=position_ids
    )
    model.model.decoder.embed_positions.cpu()

    if model.model.decoder.project_in is not None:
        model.model.decoder.project_in.to(device)
        inputs_embeds = model.model.decoder.project_in(inputs_embeds)
        model.model.decoder.project_in.cpu()

    hidden_states = inputs_embeds + pos_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # check if head_mask has a correct number of layers specified if desired
    for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
        if attn_mask is not None:
            if attn_mask.size()[0] != (len(model.model.decoder.layers)):
                raise ValueError(
                    f"The `{mask_name}` should be specified for {len(model.model.decoder.layers)} layers, but it is for"
                    f" {head_mask.size()[0]}."
                )

    for idx, decoder_layer in enumerate(model.model.decoder.layers):
        # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = (
            past_key_values[idx] if past_key_values is not None else None
        )

        decoder_layer.to(device)
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=causal_attention_mask,
            position_ids=position_ids,
            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        decoder_layer.cpu()

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (
                layer_outputs[2 if output_attentions else 1],
            )

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm.to(device)
        hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        model.model.decoder.final_layer_norm.cpu()

    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out.to(device)
        hidden_states = model.model.decoder.project_out(hidden_states)
        model.model.decoder.project_out.cpu()

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        outputs = tuple(
            v
            for v in [
                hidden_states,
                next_cache,
                all_hidden_states,
                all_self_attns,
            ]
            if v is not None
        )
    else:
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    model.lm_head.to(device)
    logits = model.lm_head(outputs[0]).contiguous()
    model.lm_head.cpu()
    loss = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
