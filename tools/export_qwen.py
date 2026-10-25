#!/usr/bin/env python3
"""
Export a Hugging Face Qwen2 checkpoint into micro-trt's weight format.

Usage:  python3 tools/export_qwen.py [model_id] [out_prefix]
        defaults: Qwen/Qwen2.5-0.5B  models/qwen

Writes <out_prefix>.bin and <out_prefix>.idx (format in mtrt_format.py).

Layout decisions, made here once so the engine never has to think about them:

  * Linear weights are TRANSPOSED. PyTorch stores nn.Linear weights as
    (out x in) and computes x @ W.T; micro-trt's GEMM computes x @ W with W as
    (in x out). Transposing at export time means every projection in the engine
    is a plain gemm(x, W) -- the same convention the MLP stage already uses.

  * The embedding table is NOT transposed. It is a lookup table indexed by
    token id, so row t must be the vector for token t.

  * The output projection is written separately as lm_head (hidden x vocab),
    even though Qwen2.5-0.5B ties it to the embedding. That costs ~545 MB of
    extra GPU memory in fp32, and buys a logits computation that is one more
    ordinary GEMM instead of a special transposed case.
"""

import os
import sys

import numpy as np

from mtrt_format import Writer

try:
    import torch
    from transformers import AutoModelForCausalLM
except ImportError:
    sys.exit("Needs PyTorch and transformers. Run: pip install torch transformers")


def rope_theta(cfg):
    # transformers 5 moved this into a rope_parameters dict; 4.x has it flat.
    params = getattr(cfg, "rope_parameters", None) or {}
    if "rope_theta" in params:
        return params["rope_theta"]
    return cfg.rope_theta


def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "models/qwen"
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    print(f"Loading {model_id} in float32 ...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    cfg = model.config
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    if cfg.model_type != "qwen2":
        sys.exit(f"Expected a qwen2 model, got model_type={cfg.model_type!r}")

    head_dim = cfg.hidden_size // cfg.num_attention_heads

    w = Writer(prefix)
    w.config("hidden", cfg.hidden_size)
    w.config("layers", cfg.num_hidden_layers)
    w.config("heads", cfg.num_attention_heads)
    w.config("kv_heads", cfg.num_key_value_heads)
    w.config("head_dim", head_dim)
    w.config("intermediate", cfg.intermediate_size)
    w.config("vocab", cfg.vocab_size)
    w.config("rope_theta", rope_theta(cfg))
    w.config("rms_eps", cfg.rms_norm_eps)

    def linear(name, key, bias=False):
        w.tensor(name, sd[key + ".weight"].T)  # (out x in) -> (in x out)
        if bias:
            w.tensor(name.replace(".w", ".b", 1), sd[key + ".bias"])

    w.tensor("embed", sd["model.embed_tokens.weight"])

    for i in range(cfg.num_hidden_layers):
        hf = f"model.layers.{i}"
        L = f"L{i}"
        w.tensor(f"{L}.attn_norm", sd[f"{hf}.input_layernorm.weight"])
        # Qwen2 has biases on q, k and v -- but not on o. Llama has none at all.
        linear(f"{L}.wq", f"{hf}.self_attn.q_proj", bias=True)
        linear(f"{L}.wk", f"{hf}.self_attn.k_proj", bias=True)
        linear(f"{L}.wv", f"{hf}.self_attn.v_proj", bias=True)
        linear(f"{L}.wo", f"{hf}.self_attn.o_proj")
        w.tensor(f"{L}.mlp_norm", sd[f"{hf}.post_attention_layernorm.weight"])
        linear(f"{L}.w_gate", f"{hf}.mlp.gate_proj")
        linear(f"{L}.w_up", f"{hf}.mlp.up_proj")
        linear(f"{L}.w_down", f"{hf}.mlp.down_proj")

    w.tensor("final_norm", sd["model.norm.weight"])
    lm_head = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
    w.tensor("lm_head", lm_head.T)
    w.close()

    n_params = sum(v.size for k, v in sd.items() if k != "lm_head.weight")
    size_gb = os.path.getsize(prefix + ".bin") / 1e9
    print(f"Wrote {prefix}.bin ({size_gb:.2f} GB) and {prefix}.idx")
    print(f"  {n_params / 1e6:.1f}M parameters, {cfg.num_hidden_layers} layers, "
          f"hidden {cfg.hidden_size}, {cfg.num_attention_heads} query heads / "
          f"{cfg.num_key_value_heads} KV heads, head_dim {head_dim}")


if __name__ == "__main__":
    main()
