"""Train the learned Ijaz student: CAMeLBERT-MSA tags every word with the teacher's deletion level.

Labels per word: 0 keep, 1/2/3 deletion level (see data/teacher_prompt.txt).
5% of passages are held out to measure how often the student agrees with the teacher.

Run: python scripts/train_student.py      (reads data/teacher_labels.jsonl, saves models/ijaz-student)
"""
import argparse
import json
import os
import random
import sys

import torch
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, get_linear_schedule_with_warmup)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.teacher_label import word_levels  # noqa: E402

BASE = "CAMeL-Lab/bert-base-arabic-camelbert-msa"


def encode(tok, words, levels):
    """Label only the first subword of each word; the rest are ignored (-100)."""
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=512)
    labels, prev = [], None
    for w in enc.word_ids():
        labels.append(-100 if w is None or w == prev else levels[w])
        prev = w
    enc["labels"] = labels
    return dict(enc)


def short_pieces(words, levels, lo=40, hi=90):
    """Cut a labelled passage into 40-90-word pieces at sentence ends, keeping each word's label,
    so the student also sees short passages (most ARCD passages are under 120 words)."""
    pieces, start, last_end = [], 0, None
    for i, w in enumerate(words):
        if w.endswith((".", "؟", "!", "?")):
            last_end = i + 1
            if last_end - start >= lo:
                pieces.append((start, last_end))
                start = last_end
        if i + 1 - start >= hi:  # no sentence end in sight: cut here
            pieces.append((start, i + 1))
            start = i + 1
    return [(words[a:b], levels[a:b]) for a, b in pieces if b - a >= lo // 2]


def evaluate(model, batches, dev):
    """Word-level agreement with the teacher: exact level, and keep-vs-delete."""
    model.eval()
    conf = [[0] * 4 for _ in range(4)]
    with torch.no_grad():
        for b in batches:
            b = {k: v.to(dev) for k, v in b.items()}
            pred = model(**{k: v for k, v in b.items() if k != "labels"}).logits.argmax(-1)
            for p, g in zip(pred[b["labels"] != -100].tolist(), b["labels"][b["labels"] != -100].tolist()):
                conf[g][p] += 1
    model.train()
    total = sum(map(sum, conf))
    exact = sum(conf[i][i] for i in range(4)) / total
    binary = (conf[0][0] + sum(conf[g][p] for g in range(1, 4) for p in range(1, 4))) / total
    return exact, binary, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/teacher_labels.jsonl")
    ap.add_argument("--out", default="models/ijaz-student")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--short", action="store_true", help="also train on 40-90-word pieces of each passage")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    random.Random(7).shuffle(rows)
    n_val = max(1, len(rows) // 20)
    print(f"{len(rows)} labelled passages: {len(rows) - n_val} train, {n_val} held out")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=4).to(dev)
    data = [encode(tok, r["words"], word_levels(r["spans"], len(r["words"]))) for r in rows]
    collate = DataCollatorForTokenClassification(tok)
    batches = lambda d: [collate(d[i:i + args.batch]) for i in range(0, len(d), args.batch)]  # noqa: E731
    val = batches(data[:n_val])
    train = data[n_val:]
    if args.short:  # pieces only from training passages, so held-out stays unseen
        for r in rows[n_val:]:
            for w, lv in short_pieces(r["words"], word_levels(r["spans"], len(r["words"]))):
                train.append(encode(tok, w, lv))
        print(f"with short pieces: {len(train)} training examples")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    best = -1
    for epoch in range(1, args.epochs + 1):
        random.Random(epoch).shuffle(train)
        loss_sum = 0.0
        for b in batches(train):
            loss = model(**{k: v.to(dev) for k, v in b.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += loss.item()
        exact, binary, conf = evaluate(model, val, dev)
        print(f"epoch {epoch}: train loss {loss_sum / len(batches(train)):.3f} | held-out agreement "
              f"exact level {exact:.1%}, keep-vs-delete {binary:.1%}")
        if binary > best:
            best = binary
            model.save_pretrained(args.out)
            tok.save_pretrained(args.out)
    print("confusion on held-out, last epoch (rows = teacher level 0-3, cols = student):")
    for g, row in enumerate(conf):
        print(f"  {g}: {row}")
    print(f"Saved best model (keep-vs-delete {best:.1%}) to {args.out}")


if __name__ == "__main__":
    main()
