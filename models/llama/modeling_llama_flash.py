from typing import Optional, Tuple

import torch
import transformers
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def llama_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions is True:
        raise ValueError("output_attentions=True can not be supported with BetterTransformer.")

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0)
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [
            torch.nn.functional.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [
            torch.nn.functional.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [
            torch.nn.functional.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)
        ]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    # repeat k/v heads if n_kv_heads < n_heads
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    if bsz == 1 or self.training:
        # BEWARE: at this stage, attention_mask is not the same as in transformers llama
        if query_states.shape[2] > 1:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=True
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states, attn_mask=None, dropout_p=0.0, is_causal=False
            )
    else:
        # At this stage, **attention_mask is the same** as in transformers llama
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum(
            [torch.nn.functional.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)]
        )
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # [bsz, seq_len]
    return attention_mask


def replace_llama_attn_with_flash_attn():
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask
    )
    transformers.models.llama.modeling_llama.LlamaAttention.forward = llama_forward


def load_llama_flash_class() -> type(LlamaForCausalLM):
    replace_llama_attn_with_flash_attn()
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    return LlamaForCausalLM


if __name__ == '__main__':
    # def test_llama_generate():
    from transformers import LlamaTokenizer, LlamaForCausalLM

    model_name_or_path = "/mnt/h/models/chinese-alpaca-2-7b"
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    input_ids = tokenizer("Hello, my dog is cute", return_tensors="pt").input_ids

    with torch.no_grad():
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
        output = model.generate(input_ids, do_sample=False, max_length=20)
        print(tokenizer.batch_decode(output, skip_special_tokens=True))
        del model

        replace_llama_attn_with_flash_attn()
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, device_map='auto')
        output = model.generate(input_ids, do_sample=False, max_length=20)
        print(tokenizer.batch_decode(output, skip_special_tokens=True))
