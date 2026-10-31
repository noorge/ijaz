"""Train the "answerness" signal: how likely a word is to be part of an answer people ask about.

Data: TyDiQA-GoldP Arabic (human questions on Arabic Wikipedia, answer spans marked by people).
Every word inside an answer span = 1, else 0. Ijaz stays question-free at compression time:
questions are only used here, to learn which kinds of words carry askable facts.
Articles and passages that appear in ARCD (our eval set) are excluded.

Run: python scripts/train_answerness.py      (saves models/ijaz-answerness)
"""
import argparse
import json
import os
import random
import re
import sys

import torch
from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, get_linear_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_student import BASE, encode  # noqa: E402
from ijaz.normalize import normalize_for_scoring  # noqa: E402

WINDOW = 200  # same 200-word windows the compressor reads


def load_examples():
    ds = load_dataset("google-research-datasets/tydiqa", "secondary_task", split="train")
    arcd = json.load(open("data/arcd.json", encoding="utf-8"))["data"]
    arcd_titles = {a["title"] for a in arcd}
    arcd_starts = {normalize_for_scoring(p["context"])[:80] for a in arcd for p in a["paragraphs"]}
    spans, skipped = {}, 0
    for x in ds:
        if not x["id"].startswith("arabic"):
            continue
        if x["title"] in arcd_titles or normalize_for_scoring(x["context"])[:80] in arcd_starts:
            skipped += 1
            continue
        for s, t in zip(x["answers"]["answer_start"], x["answers"]["text"]):
            spans.setdefault(x["context"], set()).add((s, s + len(t)))
    print(f"{len(spans)} Arabic passages ({skipped} questions skipped: article or passage in ARCD)")
    out = []
    for ctx, sp in spans.items():
        words, labels = [], []
        for m in re.finditer(r"\S+", ctx):
            words.append(m.group())
            labels.append(int(any(a < m.end() and m.start() < b for a, b in sp)))
        for i in range(0, len(words), WINDOW):
            if any(labels[i:i + WINDOW]):  # windows with no answer teach nothing useful
                out.append((words[i:i + WINDOW], labels[i:i + WINDOW]))
    return out


def auc(pos, neg, k=20000):
    """Chance a random answer word scores above a random other word (0.5 = guessing)."""
    rng = random.Random(0)
    pairs = [(rng.choice(pos), rng.choice(neg)) for _ in range(k)]
    return sum((p > n) + 0.5 * (p == n) for p, n in pairs) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/ijaz-answerness")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    examples = load_examples()
    random.Random(7).shuffle(examples)
    n_val = len(examples) // 20
    print(f"{len(examples)} windows: {len(examples) - n_val} train, {n_val} held out")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=2).to(dev)
    data = [encode(tok, w, lv) for w, lv in examples]
    collate = DataCollatorForTokenClassification(tok)
    batches = lambda d: [collate(d[i:i + args.batch]) for i in range(0, len(d), args.batch)]  # noqa: E731
    val, train = batches(data[:n_val]), data[n_val:]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1
    for epoch in range(1, args.epochs + 1):
        random.Random(epoch).shuffle(train)
        model.train()
        for b in batches(train):
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
        model.eval()
        pos, neg = [], []
        with torch.no_grad():
            for b in val:
                b = {k: v.to(dev) for k, v in b.items()}
                p = model(**{k: v for k, v in b.items() if k != "labels"}).logits.softmax(-1)[..., 1]
                m = b["labels"] != -100
                for pr, g in zip(p[m].tolist(), b["labels"][m].tolist()):
                    (pos if g else neg).append(pr)
        a = auc(pos, neg)
        print(f"epoch {epoch}: held-out AUC {a:.3f} (answer words {len(pos)}, others {len(neg)})", flush=True)
        if a > best:
            best = a
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    print(f"Saved best (AUC {best:.3f}) to {args.out}")


if __name__ == "__main__":
    main()
