"""Teacher labels for the learned Ijaz: which word spans can be deleted, at which level, and why.

The teacher reads data/teacher_prompt.txt (classical Arabic rules of الحذف) and a passage with
numbered words, and returns spans {from, to, level, reason}. A small student model learns from these.

Passages come from Arabic Wikipedia, skipping every article that appears in ARCD (our eval set).

Pilot (compare teachers on a few passages, then open results/teacher_pilot.html):
  python scripts/teacher_label.py --n 5 --teachers openai allam
ALLaM runs locally through llama-server (OpenAI-compatible) on http://127.0.0.1:8080/v1.
"""
import argparse
import html
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

from ijaz.learned import never_drop as _never, protected_indices  # noqa: E402  same rule as the student

load_dotenv()

REASONS = ["حشو", "تطويل", "تكرار", "توكيد", "تذييل", "إيغال", "إبهام", "دليل",
           "إسناد", "شاهد", "استطراد", "اعتراض", "فضلة"]
SCHEMA = {
    "type": "object",
    "properties": {"spans": {"type": "array", "items": {
        "type": "object",
        "properties": {"from": {"type": "integer"}, "to": {"type": "integer"},
                       "level": {"type": "integer", "enum": [1, 2, 3]},
                       "reason": {"type": "string", "enum": REASONS}},
        "required": ["from", "to", "level", "reason"], "additionalProperties": False}}},
    "required": ["spans"], "additionalProperties": False,
}


# ---------------------------------------------------------------- passages
def load_passages(n, seed=7, path="data/train_passages.jsonl"):
    """Arabic Wikipedia paragraphs of 80-200 words, one per article, no ARCD articles. Cached."""
    cached = [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []
    if len(cached) >= n:
        return cached[:n]
    from datasets import load_dataset
    arcd_titles = {a["title"] for a in json.load(open("data/arcd.json", encoding="utf-8"))["data"]}
    rng = random.Random(seed)
    seen = {p["id"] for p in cached}
    stream = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)
    with open(path, "a", encoding="utf-8") as f:
        for art in stream:
            if len(cached) >= n:
                break
            if art["title"] in arcd_titles or art["id"] in seen or rng.random() > 0.05:
                continue  # skip ARCD articles; sample ~5% of articles for variety
            paras = [p.strip() for p in art["text"].split("\n")
                     if 80 <= len(p.split()) <= 200 and not p.strip().endswith(":")]
            if not paras:
                continue
            p = {"id": art["id"], "title": art["title"], "text": rng.choice(paras)}
            cached.append(p)
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    return cached[:n]


# ---------------------------------------------------------------- teachers
def ask(teacher, prompt):
    from openai import OpenAI
    if teacher == "allam":
        client, model = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local"), "allam"
        fmt = {"type": "json_schema", "json_schema": {"name": "spans", "schema": SCHEMA}}
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           response_format=fmt, temperature=0, max_tokens=2000)
    else:
        client, model = OpenAI(), os.getenv("TEACHER_MODEL", os.getenv("OPENAI_MODEL", "gpt-5.4-mini"))
        fmt = {"type": "json_schema", "json_schema": {"name": "spans", "schema": SCHEMA, "strict": True}}
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                           response_format=fmt, max_completion_tokens=8000)
    return model, r.choices[0].message.content, r.usage.prompt_tokens, r.usage.completion_tokens


def clean_spans(spans, n_words):
    """Drop out-of-range and overlapping spans. Returns (spans, number dropped)."""
    good, taken = [], set()
    for s in sorted(spans, key=lambda s: s["from"]):
        rng = set(range(s["from"], s["to"] + 1))
        if s["from"] < 0 or s["to"] >= n_words or s["from"] > s["to"] or rng & taken:
            continue
        good.append(s)
        taken |= rng
    return good, len(spans) - len(good)


def word_levels(spans, n_words):
    """Per word: 0 = keep (المتن), else the deletion level 1-3."""
    lv = [0] * n_words
    for s in spans:
        for i in range(s["from"], s["to"] + 1):
            lv[i] = s["level"]
    return lv


def violations(words, lv):
    """Deleted words the prompt says must never go: negation/exception/condition/contrast, digits."""
    return [w for w, l in zip(words, lv) if l and _never(w)]


def enforce(spans, words):
    """Hard rules in code, not just in the prompt: cut never-delete words out of every span,
    plus the word right after each (its scope: لم [تتطور], 27 [محافظة])."""
    keep = protected_indices(words)
    out = []
    for s in spans:
        start = None
        for i in range(s["from"], s["to"] + 2):
            ok = i <= s["to"] and i not in keep
            if ok and start is None:
                start = i
            elif not ok and start is not None:
                out.append(dict(s, **{"from": start, "to": i - 1}))
                start = None
    return out


# ---------------------------------------------------------------- report
COLORS = {1: "#f4b6b6", 2: "#f7d9a8", 3: "#c9d8f5"}


def render(rows, path="results/teacher_pilot.html"):
    out = ["<html dir='rtl'><meta charset='utf-8'><body style='font-family:Tahoma;max-width:900px;margin:auto;line-height:2'>",
           "<h2>Teacher pilot</h2><p>Highlighted = deleted. "
           + " ".join(f"<span style='background:{c}'>level {k}</span>" for k, c in COLORS.items())
           + ". Hover a word to see the reason.</p>"]
    for r in rows:
        words, lv = r["words"], word_levels(r["spans"], len(r["words"]))
        reason = {}
        for s in r["spans"]:
            for i in range(s["from"], s["to"] + 1):
                reason[i] = s["reason"]
        out.append(f"<hr><h3>{html.escape(r['title'])} — {r['teacher']} ({r['model']})</h3>")
        out.append(f"<p dir='ltr'>words {len(words)} | spans {len(r['spans'])} | bad spans dropped {r['bad_spans']}"
                   f" | rule violations: {html.escape(' '.join(r['violations'])) or 'none'}</p><p>")
        for i, w in enumerate(words):
            t = html.escape(w)
            out.append(f"<span title='{reason[i]} (L{lv[i]})' style='background:{COLORS[lv[i]]}'>{t}</span> "
                       if lv[i] else t + " ")
        out.append("</p>")
        for k in (1, 2, 3):
            kept = [w for w, l in zip(words, lv) if not l or l > k]
            out.append(f"<p><b dir='ltr'>after level ≤{k} ({len(kept)}/{len(words)} words):</b> {html.escape(' '.join(kept))}</p>")
    out.append("</body></html>")
    open(path, "w", encoding="utf-8").write("\n".join(out))
    return path


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--teachers", nargs="+", default=["openai", "allam"])
    ap.add_argument("--out", default="data/teacher_pilot.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    template = open("data/teacher_prompt.txt", encoding="utf-8").read()
    passages = load_passages(args.n)
    print(f"{len(passages)} passages | teachers {args.teachers}")

    # resumable: every finished passage is saved at once, and a rerun skips what is already saved
    rows = [json.loads(l) for l in open(args.out, encoding="utf-8")] if os.path.exists(args.out) else []
    done = {(r["id"], r["teacher"]) for r in rows}
    stop = threading.Event()  # set when the API says we are out of credits
    lock = threading.Lock()
    out = open(args.out, "a", encoding="utf-8")

    def label(job):
        p, teacher = job
        if stop.is_set():
            return None
        words = p["text"].split()
        numbered = " ".join(f"[{i}]{w}" for i, w in enumerate(words))
        try:
            model, raw, tin, tout = ask(teacher, template.replace("{passage}", numbered))
            spans, bad = clean_spans(json.loads(raw)["spans"], len(words))
        except Exception as ex:
            if "insufficient_quota" in str(ex) or "credit" in str(ex).lower():
                stop.set()
            print(f"  {teacher} failed on {p['id']}: {ex.__class__.__name__}: {str(ex)[:150]}")
            return None
        viol = violations(words, word_levels(spans, len(words)))  # teacher mistakes, before the fix-up
        spans = enforce(spans, words)
        row = {"id": p["id"], "title": p["title"], "teacher": teacher, "model": model, "words": words,
               "spans": spans, "bad_spans": bad, "violations": viol,
               "tokens_in": tin, "tokens_out": tout}
        with lock:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            new = len(rows) - len(done)
            if new % 50 == 0:
                print(f"  {new} new labelled ({len(rows)} total)")
        return row

    jobs = [(p, t) for t in args.teachers for p in passages if (p["id"], t) not in done]
    print(f"To label: {len(jobs)} (already done: {len(done)})")
    for t in args.teachers:
        tjobs = [j for j in jobs if j[1] == t]
        # ALLaM runs on one local GPU: one request at a time
        with ThreadPoolExecutor(1 if t == "allam" else args.workers) as ex:
            list(ex.map(label, tjobs))
    out.close()
    if stop.is_set():
        print("Stopped: out of API credits. Top up, then rerun the same command to continue.")
    for t in args.teachers:
        rs = [r for r in rows if r["teacher"] == t]
        if not rs:
            continue
        nw = sum(len(r["words"]) for r in rs)
        lv = [l for r in rs for l in word_levels(r["spans"], len(r["words"]))]
        print(f"{t}: {len(rs)} ok | deleted L1 {lv.count(1)/nw:.0%} L2 {lv.count(2)/nw:.0%} L3 {lv.count(3)/nw:.0%}"
              f" | bad spans {sum(r['bad_spans'] for r in rs)} | violations {sum(len(r['violations']) for r in rs)}"
              f" | tokens in/out {sum(r['tokens_in'] for r in rs)}/{sum(r['tokens_out'] for r in rs)}")
    print("Report:", render(rows[:20]))


if __name__ == "__main__":
    main()
