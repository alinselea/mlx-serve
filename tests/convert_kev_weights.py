#!/usr/bin/env python3
"""Package a Kev checkpoint (github.com/jaredpalmer/kev) as an mlx-serve model dir.

USER-RUN, from a kev checkout with its serve extras (mlx-lm, torch, huggingface_hub):
    cd ~/kev && uv run --extra serve python /path/to/mlx-serve/tests/convert_kev_weights.py \\
        jaredpalmer/kev-4b --out ~/.mlx-serve/models/jaredpalmer/Kev-4B-MLX-Serve-8bit --q-bits 8
Produces the dir `src/kev.zig` loads:
    <out>/config.json, model*.safetensors, tokenizer files   the base Qwen3.5 text model with the LoRA MERGED
    <out>/kev_head.safetensors   pointer head: q.weight [P,H], q.bias [P], k.weight [P,H], k.bias [P], f32
    <out>/kev_config.json        {"format": "kev", "format_version": 1, "head_dim", "temperature", "delimiters", "source"}

WHY EACH STEP EXISTS
(a) NO PICKLE AT SERVE TIME. A Kev run ships its head as `head.pt` (a pickle, which can run code). It is read
    here once with `torch.load(weights_only=True)` and written as safetensors; the server never opens a .pt file.
(b) MERGED, SANITIZED WEIGHTS. The adapter is folded into the base exactly as kev's MLX backend does
    (`kev.mlx_model.merge_lora`: W + (B@A)·scale in fp32 on CPU, one rounding to the base dtype), on the model
    mlx-lm loaded, so the tensors are already sanitized (norm offsets, conv layout, names). mlx-lm saves them in
    the layout mlx-serve reads for `qwen3_5`. Copying raw HF tensors would skip that sanitation.
(c) ONLY WHAT THE SERVER NEEDS. The pack carries the head tensors, head_dim, the calibrated temperature, the five
    delimiter strings and {repo, revision, base, base_revision}. Nothing else from the checkpoint's metadata
    (training arguments, evaluation data, private paths) is copied.
(d) QUANTIZED BY DEFAULT. `--q-bits 8` (the default) quantizes the merged trunk with mlx-lm's own quantizer
    (affine, group 64), the layout mlx-serve's fast kernels read; `--q-bits 0` keeps bf16, the quality reference.
    The head always stays f32.
v1 supports LoRA checkpoints on the hybrid Qwen3.5 bases, the ones kev serves on MLX. Full-weight,
option-isolation and trained-token-embedding checkpoints are refused by name.
"""
import argparse, json, os, shutil, sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from mlx_lm.utils import load_config, load_model, quantize_model, save_config, save_model

from kev.checkpoint import resolve_run
from kev.mlx_model import merge_lora
from kev.model import SPECIAL

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                   "vocab.json", "merges.txt", "chat_template.jinja")


def read_head(run_dir):
    """head.pt, loaded safely: the pointer-head state dict plus the metadata the pack keeps."""
    meta = torch.load(Path(run_dir) / "head.pt", map_location="cpu", weights_only=True)
    if meta.get("weights") not in (None, "lora"):   # older runs predate the field; the adapter file decides then
        sys.exit(f"unsupported checkpoint: weights={meta.get('weights')!r} (v1 packs LoRA checkpoints only)")
    for flag in ("option_isolation", "special_embeddings"):
        if meta.get(flag):
            sys.exit(f"unsupported checkpoint: {flag} is on (not available on kev's MLX backend)")
    head = {k: v.float().contiguous() for k, v in meta["head"].items()}
    if sorted(head) != ["k.bias", "k.weight", "q.bias", "q.weight"]:
        sys.exit(f"unexpected head tensors {sorted(head)}")
    p, h = head["q.weight"].shape
    if p != meta.get("head_dim", p) or head["k.weight"].shape != (p, h) or head["q.bias"].shape != (p,) or head["k.bias"].shape != (p,):
        sys.exit("head tensor shapes disagree with each other or with head_dim")
    t = float(meta.get("temperature", 1.0))
    if not (np.isfinite(t) and t > 0):
        sys.exit(f"temperature must be finite and positive, got {t}")
    if not all(torch.isfinite(v).all() for v in head.values()):
        sys.exit("head weights contain non-finite values")
    return head, p, t, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="kev run: a hub id (jaredpalmer/kev-4b) or a local run dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--q-bits", type=int, default=8, choices=(0, 4, 8), help="trunk quantization (0 = bf16)")
    ap.add_argument("--q-group-size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(resolve_run(args.run))   # kev's own metadata reader is not used: it does not pin weights_only
    if not (run_dir / "adapter_config.json").exists():
        sys.exit("unsupported checkpoint: no adapter_config.json (v1 packs LoRA checkpoints only)")
    head, head_dim, temperature, meta = read_head(run_dir)
    base_dir = Path(resolve_run(f"{meta['base']}@{meta.get('base_revision') or ''}"))
    base_cfg = load_config(base_dir)
    if base_cfg.get("model_type") not in ("qwen3_5", "qwen3_5_moe"):
        sys.exit(f"unsupported base {meta['base']}: model_type {base_cfg.get('model_type')!r} (v1 = hybrid Qwen3.5)")

    lm, _ = load_model(base_dir)
    merged = merge_lora(lm, run_dir, 1.0)
    hidden = lm.language_model.model.embed_tokens.weight.shape[1]
    if head["q.weight"].shape[1] != hidden:
        sys.exit(f"head input width {head['q.weight'].shape[1]} != backbone hidden size {hidden}")

    cfg = dict(base_cfg)
    if args.q_bits:
        lm, cfg = quantize_model(lm, cfg, args.q_group_size, args.q_bits)
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    save_model(out, lm, donate_model=True)
    save_config(cfg, out / "config.json")
    for name in TOKENIZER_FILES:
        src = run_dir / name if (run_dir / name).exists() else base_dir / name
        if src.exists():
            shutil.copyfile(src, out / name)
    mx.save_safetensors(str(out / "kev_head.safetensors"), {k: mx.array(v.numpy()) for k, v in head.items()})
    with open(out / "kev_config.json", "w") as f:
        json.dump({"format": "kev", "format_version": 1, "head_dim": head_dim, "temperature": temperature,
                   "delimiters": {"state": SPECIAL[0], "question": SPECIAL[1], "option": SPECIAL[2],
                                  "option_end": SPECIAL[3], "decide": SPECIAL[4]},
                   "source": {"repo": None if os.path.isdir(args.run) else args.run.partition("@")[0],
                              "revision": None if os.path.isdir(args.run) else run_dir.name,
                              "base": meta["base"], "base_revision": meta.get("base_revision")}}, f, indent=1)
    print(f"merged {merged} LoRA tensors; trunk {'bf16' if not args.q_bits else f'{args.q_bits}-bit g{args.q_group_size}'}; "
          f"head_dim {head_dim}, temperature {temperature:.6f}; wrote {out}")


if __name__ == "__main__":
    main()
