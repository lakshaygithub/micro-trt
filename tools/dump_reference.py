#!/usr/bin/env python3
"""
Record Hugging Face's intermediate activations for one prompt, so micro-trt can
be checked layer by layer instead of only at the final logits.

Usage:  python3 tools/dump_reference.py [model_id] [out_prefix]
        defaults: Qwen/Qwen2.5-0.5B  models/ref

Why this exists: a transformer is 24 layers of the same block. If only the
logits are compared, a mismatch says "something in ~200 kernel launches is
wrong" and nothing more. With every layer's output on disk, the first layer
that disagrees names the bug, and the layer-0 internals below narrow it down to
a single operation.

What is recorded (all float32, prompt length T, batch dimension dropped):

  ints   tokens                prompt token ids
  ints   generated             greedy continuation, GEN_TOKENS long
  tensor embed_out             T x hidden
  tensor L0.attn_norm_out      T x hidden          RMSNorm before attention
  tensor L0.q / L0.k / L0.v    T x (heads*head_dim) projections, before RoPE
  tensor L0.q_rope / L0.k_rope same, after RoPE
  tensor L0.attn_ctx           T x hidden          attention output, before o_proj
  tensor L0.attn_out           T x hidden          after o_proj
  tensor L0.mlp_norm_out       T x hidden
  tensor L0.mlp_out            T x hidden
  tensor L{i}.out              T x hidden          every layer's output
  tensor final_norm_out        T x hidden
  tensor logits                T x vocab
"""

import os
import sys

from mtrt_format import Writer

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
except ImportError:
    sys.exit("Needs PyTorch and transformers. Run: pip install torch transformers")

PROMPT = "The key to fast GPU kernels is"
GEN_TOKENS = 32


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "models/ref"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    # fp32 and eager attention: the reference should be the plainest possible
    # implementation, not a fused SDPA kernel with its own rounding behaviour.
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()
    cfg = model.config
    heads, kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = cfg.hidden_size // heads

    ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    T = ids.shape[1]

    captured = {}

    def keep(name, use_input=False):
        def hook(module, inputs, output):
            t = inputs[0] if use_input else output
            # Decoder layers return a tuple in older transformers versions and
            # a bare tensor in newer ones.
            if isinstance(t, tuple):
                t = t[0]
            captured[name] = t[0].detach().float().cpu()  # drop batch dim
        return hook

    layers = model.model.layers
    l0 = layers[0]
    hooks = [
        model.model.embed_tokens.register_forward_hook(keep("embed_out")),
        l0.input_layernorm.register_forward_hook(keep("L0.attn_norm_out")),
        l0.self_attn.q_proj.register_forward_hook(keep("L0.q")),
        l0.self_attn.k_proj.register_forward_hook(keep("L0.k")),
        l0.self_attn.v_proj.register_forward_hook(keep("L0.v")),
        l0.self_attn.o_proj.register_forward_hook(keep("L0.attn_ctx", use_input=True)),
        l0.self_attn.o_proj.register_forward_hook(keep("L0.attn_out")),
        l0.post_attention_layernorm.register_forward_hook(keep("L0.mlp_norm_out")),
        l0.mlp.register_forward_hook(keep("L0.mlp_out")),
        model.model.norm.register_forward_hook(keep("final_norm_out")),
    ]
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(keep(f"L{i}.out")))

    with torch.no_grad():
        logits = model(ids).logits[0].float().cpu()
        for h in hooks:
            h.remove()

        # RoPE happens inside the attention forward with no module boundary to
        # hook, so it is recomputed here from the captured projections using
        # transformers' own functions. This is the step most likely to be
        # implemented wrongly: Qwen/Llama rotate dimension i together with
        # i + head_dim/2 ("rotate_half"), not adjacent pairs (2i, 2i+1).
        pos = torch.arange(T, device=device).unsqueeze(0)
        q = captured["L0.q"].to(device).view(1, T, heads, head_dim).transpose(1, 2)
        k = captured["L0.k"].to(device).view(1, T, kv_heads, head_dim).transpose(1, 2)
        cos, sin = model.model.rotary_emb(q, pos)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        captured["L0.q_rope"] = q_rot.transpose(1, 2).reshape(T, heads * head_dim).cpu()
        captured["L0.k_rope"] = k_rot.transpose(1, 2).reshape(T, kv_heads * head_dim).cpu()

        out = model.generate(ids, max_new_tokens=GEN_TOKENS, do_sample=False)
        generated = out[0, T:].tolist()

    w = Writer(prefix)
    w.ints("tokens", ids[0].tolist())
    w.ints("generated", generated)
    for name, t in captured.items():
        w.tensor(name, t.numpy())
    w.tensor("logits", logits.numpy())
    w.close()

    print(f"Prompt   : {PROMPT!r}  ({T} tokens: {ids[0].tolist()})")
    print(f"Greedy   : {tok.decode(generated)!r}")
    print(f"Recorded : {len(captured) + 1} tensors -> {prefix}.bin / {prefix}.idx")


if __name__ == "__main__":
    main()
