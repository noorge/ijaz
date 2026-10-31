"""Does the teacher agree with itself? Relabel passages it already labelled and compare word by word.

Uses passages from the student's held-out set (same shuffle as train_student.py), so we can also see how
close the student is to the teacher's own consistency, the ceiling for any student.

Run: python scripts/teacher_consistency.py --n 100
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.teacher_label import ask, clean_spans, enforce, word_levels  # noqa: E402
from ijaz.learned import word_probs  # noqa: E402


def agree(a, b):
    """(exact level agreement, keep-vs-delete agreement) over all words."""
    n = len(a)
    return sum(x == y for x, y in zip(a, b)) / n, sum((x == 0) == (y == 0) for x, y in zip(a, b)) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--labels", default="data/teacher_labels.jsonl")
    ap.add_argument("--out", default="data/teacher_relabel.jsonl")
    ap.add_argument("--student", default="models/ijaz-student-5k")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    random.Random(7).shuffle(rows)  # same shuffle as train_student.py: the first 5% are held out
    held = rows[: max(1, len(rows) // 20)][: args.n]
    template = open("data/teacher_prompt.txt", encoding="utf-8").read()

    done = {}
    if os.path.exists(args.out):  # resumable
        done = {r["id"]: r for r in map(json.loads, open(args.out, encoding="utf-8"))}

    def relabel(r):
        if r["id"] in done:
            return done[r["id"]]
        words = r["words"]
        numbered = " ".join(f"[{i}]{w}" for i, w in enumerate(words))
        try:
            _, raw, _, _ = ask("openai", template.replace("{passage}", numbered))
            spans, _ = clean_spans(json.loads(raw)["spans"], len(words))
        except Exception as ex:
            print(f"  failed on {r['id']}: {ex.__class__.__name__}: {str(ex)[:120]}")
            return None
        return {"id": r["id"], "spans": enforce(spans, words)}

    with ThreadPoolExecutor(8) as ex:
        second = list(ex.map(relabel, held))
    with open(args.out, "w", encoding="utf-8") as f:
        for s in second:
            if s:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    t1, t2, st = [], [], []
    for r, s in zip(held, second):
        if not s:
            continue
        n = len(r["words"])
        t1 += word_levels(r["spans"], n)
        t2 += word_levels(s["spans"], n)
        st += [max(range(4), key=lambda k: p[k]) for p in word_probs(r["words"], model_dir=args.student)]
    print(f"{sum(1 for s in second if s)} passages, {len(t1)} words")
    for name, a, b in [("teacher run 1 vs teacher run 2", t1, t2),
                       ("student vs teacher run 1", st, t1),
                       ("student vs teacher run 2", st, t2)]:
        e, kd = agree(a, b)
        print(f"  {name:32} exact level {e:.1%} | keep-vs-delete {kd:.1%}")
    print(f"  (always-delete baseline vs run 1: keep-vs-delete {sum(x != 0 for x in t1) / len(t1):.1%})")


if __name__ == "__main__":
    main()
