"""Causal language-model pretraining for Khufu 1.0.

Put UTF-8 Arabic and English text files in data/corpus/ (or pass --corpus-dir).
This trains next-character prediction, so the model learns language patterns instead
of only memorizing question/answer pairs. Fine-tune the resulting checkpoint with
train_model.py --resume afterward.
"""

import argparse
import glob
import os
import random
import time

import numpy as np
import torch

from checkpoint import load_checkpoint, save_checkpoint
from model import Config, MiniGPT, PRESETS
from tokenizer import CharTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Khufu 1.0 Arabic/English language pretraining")
    p.add_argument("--corpus-dir", default=os.path.join(HERE, "data", "corpus"))
    p.add_argument("--out", default=os.path.join(HERE, "khufu_1_0_ctx1024.npz"))
    p.add_argument("--size", choices=list(PRESETS), default="tiny")
    p.add_argument("--block-size", type=int, default=1024,
                   help="Context length in characters for the current character tokenizer")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--ram-mb", type=int, default=450,
                   help="Approximate attention-memory budget; leave room for OS and optimizer")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_corpus(folder):
    paths = sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True))
    docs = []
    for path in paths:
        try:
            with open(path, encoding="utf-8") as f:
                value = f.read().strip()
            if len(value) >= 100:
                docs.append(value)
        except (OSError, UnicodeError) as exc:
            print(f"Skipping {path}: {exc}")
    if not docs:
        raise SystemExit(
            "No usable .txt corpus found. Add original or properly licensed Arabic/English "
            "UTF-8 text files under data/corpus/."
        )
    return "\n\n".join(docs), paths


def sample_batch(ids, batch_size, block_size, device, rng):
    starts = rng.integers(0, len(ids) - block_size - 1, size=batch_size)
    x = np.stack([ids[i:i + block_size] for i in starts])
    y = np.stack([ids[i + 1:i + block_size + 1] for i in starts])
    return (torch.tensor(x, dtype=torch.long, device=device),
            torch.tensor(y, dtype=torch.long, device=device))


def evaluate(model, ids, block_size, device, rng, batches=8, batch_size=2):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(ids, min(batch_size, 2), block_size, device, rng)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    args = parse_args()
    if args.block_size < 64 or args.block_size > 4096:
        raise SystemExit("--block-size must be between 64 and 4096 characters")
    if args.steps < 1 or args.batch_size < 1:
        raise SystemExit("--steps and --batch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus, paths = load_corpus(args.corpus_dir)
    split = max(args.block_size + 2, int(len(corpus) * 0.95))
    if len(corpus) < args.block_size * 2:
        raise SystemExit("Corpus is too small for this context length; add more text first.")

    ckpt = load_checkpoint(args.out) if args.resume and os.path.exists(args.out) else None
    if ckpt:
        tok = CharTokenizer(ckpt["tokens"])
        tok.add_chars_from(corpus)
        cfg = Config(**ckpt["config"])
        if cfg.block_size != args.block_size:
            raise SystemExit(
                f"Checkpoint context is {cfg.block_size}; requested {args.block_size}. "
                "Use the checkpoint's size or start a new output file."
            )
        cfg.vocab_size = tok.vocab_size
        start_step = int(ckpt.get("step", 0))
    else:
        tok = CharTokenizer()
        tok.add_chars_from(corpus)
        cfg = Config(block_size=args.block_size, vocab_size=tok.vocab_size,
                     **PRESETS[args.size])
        start_step = 0

    all_ids = np.asarray(tok.encode(corpus), dtype=np.int64)
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]
    if len(val_ids) <= cfg.block_size + 1:
        val_ids = train_ids[-(cfg.block_size + 2):]
    if len(train_ids) <= cfg.block_size + 1:
        raise SystemExit("Not enough training characters after tokenization.")

    model = MiniGPT(cfg).to(device)
    if ckpt:
        state = {k: torch.from_numpy(np.array(v)) for k, v in ckpt["weights"].items()}
        old_vocab = state["tok_emb.weight"].shape[0]
        if cfg.vocab_size > old_vocab:
            emb = model.tok_emb.weight.detach().clone()
            emb[:old_vocab] = state["tok_emb.weight"]
            state["tok_emb.weight"] = emb
            state["lm_head.weight"] = emb
        model.load_state_dict(state, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(args.seed + start_step)
    # Attention memory grows with batch * context^2 * heads * layers.
    bytes_per_cell = 24 * cfg.n_head * cfg.n_layer
    budget_bytes = max(64, args.ram_mb) * 1024 * 1024
    safe_batch = max(1, budget_bytes // max(1, bytes_per_cell * cfg.block_size * cfg.block_size))
    batch_size = max(1, min(args.batch_size, int(safe_batch)))

    print(f"Khufu 1.0 | device={device} | params={model.num_params():,}")
    print(f"Corpus files={len(paths)} | chars={len(all_ids):,} | vocab={tok.vocab_size}")
    print(f"Context={cfg.block_size} characters | batch={batch_size} | start_step={start_step}")

    model.train()
    step = start_step
    end_step = start_step + args.steps
    started = time.time()
    while step < end_step:
        x, y = sample_batch(train_ids, batch_size, cfg.block_size, device, rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % args.eval_every == 0 or step == end_step:
            val_loss = evaluate(model, val_ids, cfg.block_size, device, rng,
                                batch_size=batch_size)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f} "
                  f"elapsed_min={(time.time() - started) / 60:.1f}", flush=True)

        if step % args.save_every == 0 or step == end_step:
            weights = {k: v.detach().cpu().numpy()
                       for k, v in model.state_dict().items()}
            save_checkpoint(args.out, weights, cfg.to_dict(), tok.tokens, step,
                            {"pretraining": True, "corpus_chars": len(all_ids),
                             "validation_loss": val_loss if 'val_loss' in locals() else None})
            print(f"checkpoint saved: {args.out}", flush=True)

    print("Pretraining finished. Next: python train_model.py --resume --out " + args.out)


if __name__ == "__main__":
    main()
