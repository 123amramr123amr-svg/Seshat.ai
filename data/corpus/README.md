# seshat language corpus

Put UTF-8 plain-text `.txt` files in this folder (subfolders are supported).

## What to collect

- Modern Standard Arabic and Egyptian Arabic, with correct spelling and punctuation.
- High-quality English prose.
- Explanations, dialogue, science, history, mathematics, and programming concepts.
- Parallel Arabic/English material written or licensed for this project.

Use only text you created, public-domain material, or material whose license explicitly allows model training. Do not scrape private messages, personal data, or copyrighted books/articles without permission.

## Train

From the repository root:

```bash
python pretrain.py --size tiny --block-size 512 --steps 3000 --batch-size 2 --ram-mb 450
python train_model.py --resume --out seshat_chet_model.npz --size tiny --steps 1500
```

The current tokenizer is character-based, so `512` means approximately 512 characters, not 512 words or subword tokens. More context increases memory use roughly quadratically in attention. The training script saves checkpoints and can resume with `--resume`.

A small corpus is not enough to create a general-purpose assistant. Grow the licensed corpus, evaluate on held-out text, and compare validation loss before increasing model size.
