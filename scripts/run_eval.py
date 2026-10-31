"""Does the model still answer correctly after compression?

For each sampled ARCD question: compress the passage with each method at each keep rate,
ask each model, and save the answer + real billed tokens. Resumable: rerunning skips
anything already saved, so a crash never costs you money twice.

Quick test (cheap):  python scripts/run_eval.py --n 20 --models openai
Full run:            python scripts/run_eval.py --n 200
Then:                python scripts/summarize.py
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

from ijaz import compress  # noqa: E402
from ijaz.baselines import random_drop, truncate, llmlingua2  # noqa: E402
from ijaz.lexicon import NEGATIONS  # noqa: E402
from ijaz.tokens import count_tokens  # noqa: E402

load_dotenv()

PROMPT_CTX = ("اقرأ النص ثم أجب عن السؤال اعتمادا على النص فقط. "
              "اكتب الإجابة القصيرة فقط كما وردت في النص، دون شرح.\n\n"
              "النص:\n{context}\n\nالسؤال: {question}\nالإجابة:")
PROMPT_NOCTX = ("أجب عن السؤال التالي. اكتب الإجابة القصيرة فقط، دون شرح.\n\n"
                "السؤال: {question}\nالإجابة:")


# ---------------------------------------------------------------- data
def load_arcd(path="data/arcd.json"):
    if not os.path.exists(path):
        import urllib.request
        os.makedirs("data", exist_ok=True)
        url = "https://raw.githubusercontent.com/husseinmozannar/SOQAL/master/data/arcd.json"
        print("Downloading ARCD...")
        urllib.request.urlretrieve(url, path)
    data = json.load(open(path, encoding="utf-8"))
    items = []
    for art in data["data"]:
        for p in art["paragraphs"]:
            ctx = p["context"]
            for qa in p["qas"]:
                ans = qa["answers"][0]
                before = ctx[max(0, ans["answer_start"] - 60): ans["answer_start"]].split()
                items.append({
                    "id": qa["id"], "context": ctx, "question": qa["question"].strip(" -"),
                    "answers": list({a["text"] for a in qa["answers"]}),
                    "probe_number": bool(re.search(r"[0-9٠-٩]", ans["text"])),
                    "probe_negation": any(w in NEGATIONS for w in before[-6:]),
                })
    return items


# ---------------------------------------------------------------- methods
def make_context(method, ctx, rate):
    if method == "full":
        return ctx
    if method == "none":
        return None
    if method == "random":
        return random_drop(ctx, rate, seed=0)
    if method == "truncate":
        return truncate(ctx, rate)
    if method == "llmlingua2":
        return llmlingua2(ctx, rate)
    if method == "ijaz":
        return compress(ctx, rate).text
    if method == "ijaz_lm":
        return compress(ctx, rate, use_lm=True).text
    if method.startswith("ijaz_learned"):
        from ijaz.learned import compress_learned  # loads torch + the student only when asked
        # ijaz_learned = student trained on 1k labels; ijaz_learned_5k = models/ijaz-student-5k
        suffix = method[len("ijaz_learned"):]
        return compress_learned(ctx, rate, model_dir="models/ijaz-student" + suffix.replace("_", "-"))
    if method == "ijaz_final":  # settings live in ijaz/learned.py, shared with the demo
        from ijaz.learned import compress_final
        return compress_final(ctx, rate)
    if method in ("ijaz_ans", "ijaz_mix"):
        # tuned on dev questions only: answerness (+ the 5k teacher student for ijaz_mix), glue -0.3, soft +0.5
        from ijaz.learned import compress_learned
        if method == "ijaz_ans":
            return compress_learned(ctx, rate, teacher=False, glue=0.3, soft=0.5)
        return compress_learned(ctx, rate, model_dir="models/ijaz-student-5k", ans_weight=2.0, glue=0.3, soft=0.5)
    raise ValueError(method)


# ---------------------------------------------------------------- models
def call_model(provider, model, prompt, retries=5):
    for attempt in range(retries):
        try:
            if provider == "mock":  # free pipeline test: echoes part of the prompt
                ctx = prompt.split("النص:\n")[-1].split("\n")[0]
                return " ".join(ctx.split()[:4]), len(prompt.split()), 4
            if provider == "openai":
                from openai import OpenAI
                r = OpenAI().chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "400")))
                return (r.choices[0].message.content or "").strip(), r.usage.prompt_tokens, r.usage.completion_tokens
            if provider == "anthropic":
                import anthropic
                r = anthropic.Anthropic().messages.create(
                    model=model, max_tokens=150,
                    messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
                return text, r.usage.input_tokens, r.usage.output_tokens
        except Exception as ex:
            wait = 2 ** attempt
            print(f"  {provider} error ({ex.__class__.__name__}: {str(ex)[:120]}), retry in {wait}s")
            time.sleep(wait)
    return None, 0, 0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="number of questions")
    ap.add_argument("--rates", type=float, nargs="+", default=[0.75, 0.5, 0.33])
    ap.add_argument("--methods", nargs="+", default=["full", "none", "random", "llmlingua2", "ijaz"])
    ap.add_argument("--models", nargs="+", default=["openai", "anthropic"])
    ap.add_argument("--out", default="results/raw.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-probes", action="store_true", help="skip the extra negation questions")
    ap.add_argument("--dev", action="store_true",
                    help="tuning questions: outside the first 200 and the negation extras; saved to results/raw_dev.jsonl")
    args = ap.parse_args()

    model_names = {"mock": "mock", "openai": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
                   "anthropic": os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")}

    items = load_arcd()
    random.Random(args.seed).shuffle(items)
    sample = items[: args.n]
    if args.dev:  # never touches the test questions; no probes on dev
        sample = [it for it in items[200:] if not it["probe_negation"]][: args.n]
        args.no_probes = True
        if args.out == "results/raw.jsonl":
            args.out = "results/raw_dev.jsonl"
    if not args.no_probes:
        # every ARCD question whose answer sits next to a negation, plus our negation suite
        ids = {it["id"] for it in sample}
        sample += [dict(it, extra=True) for it in items[args.n:] if it["probe_negation"] and it["id"] not in ids]
        for p in json.load(open("data/negation_probe.json", encoding="utf-8")):
            sample.append({"id": p["id"], "context": p["context"], "question": p["question"],
                           "answers": [p["answer"]], "probe_number": False, "probe_negation": True,
                           "extra": True})
    items = sample
    print(f"{len(items)} questions | methods {args.methods} | rates {args.rates} | models {args.models}")

    # 1) compress everything first (cached), so model calls are the only slow part
    os.makedirs("results", exist_ok=True)
    cache_path = "results/compressions.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            r = json.loads(line)
            cache[(r["id"], r["method"], r["rate"])] = r["context"]
    if "llmlingua2" in args.methods:
        try:
            import llmlingua  # noqa: F401
        except ImportError:
            print("llmlingua not installed -> skipping LLMLingua-2 (pip install llmlingua)")
            args.methods.remove("llmlingua2")
    jobs = []
    with open(cache_path, "a", encoding="utf-8") as cf:
        for it in items:
            for m in args.methods:
                rates = [1.0] if m in ("full", "none") else args.rates
                for rate in rates:
                    key = (it["id"], m, rate)
                    if key not in cache:
                        cache[key] = make_context(m, it["context"], rate)
                        cf.write(json.dumps({"id": it["id"], "method": m, "rate": rate,
                                             "context": cache[key]}, ensure_ascii=False) + "\n")
                    jobs.append((it, m, rate))
    print(f"Compressions ready: {len(jobs)} contexts")

    # 2) ask the models
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["method"], r["rate"], r["provider"]))
    todo = [(it, m, rate, p) for it, m, rate in jobs for p in args.models
            if (it["id"], m, rate, p) not in done and m in args.methods]
    print(f"Model calls to make: {len(todo)} (already done: {len(done)})")

    lock = threading.Lock()

    def run(job):
        it, m, rate, p = job
        ctx = cache[(it["id"], m, rate)]
        prompt = PROMPT_NOCTX.format(question=it["question"]) if ctx is None else \
            PROMPT_CTX.format(context=ctx, question=it["question"])
        pred, in_tok, out_tok = call_model(p, model_names[p], prompt)
        if pred is None:
            return None
        return {"id": it["id"], "method": m, "rate": rate, "provider": p, "model": model_names[p],
                "question": it["question"], "answers": it["answers"], "prediction": pred,
                "context_tokens_o200k": count_tokens(ctx or ""),
                "original_tokens_o200k": count_tokens(it["context"]),
                "input_tokens": in_tok, "output_tokens": out_tok,
                "probe_number": it["probe_number"], "probe_negation": it["probe_negation"],
                "extra": it.get("extra", False)}

    with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(run, j) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            if row:
                with lock:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
            if k % 50 == 0:
                print(f"  {k}/{len(todo)}")
    print(f"Done. Results in {args.out}. Next: python scripts/summarize.py")


if __name__ == "__main__":
    main()
