"""Turn results/raw.jsonl into the numbers and chart for the video.

Metrics per model x method x keep rate:
  F1 / EM / correct%    answer quality (Arabic-normalized)
  utilization           (acc - acc_no_context) / (acc_full - acc_no_context), same metric
                        the cross-lingual audit used, so you can compare to their Arabic 0.16-0.29
  tokens/correct        real billed input tokens spent per correct answer
  cost/1k correct       if prices are set in .env
Also the probe slices: questions whose answers involve numbers or sit next to a negation.

Run: python scripts/summarize.py
"""
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

from ijaz.normalize import normalize_for_scoring  # noqa: E402

load_dotenv()


def f1(pred, gold):
    p, g = normalize_for_scoring(pred).split(), normalize_for_scoring(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def score(row):
    if row["answers"][0] in ("نعم", "لا"):  # yes/no negation suite
        gold, other = row["answers"][0], ("لا" if row["answers"][0] == "نعم" else "نعم")
        words = normalize_for_scoring(row["prediction"]).split()
        ok = gold in words[:3] and other not in words[:3]
        return float(ok), ok, ok
    best_f1 = max(f1(row["prediction"], a) for a in row["answers"])
    em = any(normalize_for_scoring(row["prediction"]) == normalize_for_scoring(a) for a in row["answers"])
    contains = any(normalize_for_scoring(a) and normalize_for_scoring(a) in normalize_for_scoring(row["prediction"])
                   for a in row["answers"])
    return best_f1, em, (best_f1 >= 0.5 or contains)


def price(provider):
    key = "OPENAI" if provider == "openai" else "CLAUDE"
    try:
        return float(os.getenv(f"{key}_PRICE_IN")), float(os.getenv(f"{key}_PRICE_OUT"))
    except (TypeError, ValueError):
        return None


def main(path="results/raw.jsonl"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    groups = defaultdict(list)
    for r in rows:
        r["f1"], r["em"], r["correct"] = score(r)
    suite = [r for r in rows if r.get("extra")]          # negation-only extras
    rows_main = [r for r in rows if not r.get("extra")]  # the random sample
    for r in rows_main:
        groups[(r["model"], r["method"], r["rate"])].append(r)

    acc = {k: sum(x["correct"] for x in v) / len(v) for k, v in groups.items()}
    table = []
    for (model, method, rate), v in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], -x[0][2])):
        n = len(v)
        a = acc[(model, method, rate)]
        full, none = acc.get((model, "full", 1.0)), acc.get((model, "none", 1.0))
        util = (a - none) / (full - none) if full is not None and none is not None and full != none else None
        n_correct = sum(x["correct"] for x in v)
        in_tok = sum(x["input_tokens"] for x in v)
        out_tok = sum(x["output_tokens"] for x in v)
        pr = price(v[0]["provider"])
        cost = (in_tok * pr[0] + out_tok * pr[1]) / 1e6 if pr else None
        probe_num = [x for x in v if x["probe_number"]]
        probe_neg = [x for x in v if x["probe_negation"]] + \
            [x for x in suite if (x["model"], x["method"], x["rate"]) == (model, method, rate)]
        table.append({
            "model": model, "method": method, "keep_rate": rate, "n": n,
            "f1": round(sum(x["f1"] for x in v) / n, 3),
            "em": round(sum(x["em"] for x in v) / n, 3),
            "correct": round(a, 3),
            "utilization": round(util, 3) if util is not None else "",
            "achieved_rate": round(sum(x["context_tokens_o200k"] for x in v) /
                                   max(1, sum(x["original_tokens_o200k"] for x in v)), 3),
            "input_tokens_per_correct": round(in_tok / n_correct, 1) if n_correct else "",
            "cost_per_1k_correct_usd": round(cost / n_correct * 1000, 3) if cost and n_correct else "",
            "numbers_correct": round(sum(x["correct"] for x in probe_num) / len(probe_num), 3) if probe_num else "",
            "negation_correct": round(sum(x["correct"] for x in probe_neg) / len(probe_neg), 3) if probe_neg else "",
            "n_numbers": len(probe_num), "n_negation": len(probe_neg),
        })

    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    cols = ["model", "method", "keep_rate", "n", "correct", "f1", "utilization", "achieved_rate",
            "input_tokens_per_correct", "numbers_correct", "negation_correct"]
    print(" | ".join(cols))
    for t in table:
        print(" | ".join(str(t[c]) for c in cols))
    print("\nSaved results/summary.csv")
    plot(table)


def plot(table):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib for the chart")
        return
    models = sorted({t["model"] for t in table})
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5), squeeze=False)
    colors = {"ijaz": "#0F766E", "llmlingua2": "#B45309", "random": "#9CA3AF",
              "truncate": "#6B7280", "ijaz_lm": "#14B8A6"}
    for ax, model in zip(axes[0], models):
        rows = [t for t in table if t["model"] == model]
        for method, color in colors.items():
            pts = sorted((t["keep_rate"], t["correct"]) for t in rows if t["method"] == method)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color, label=method, lw=2)
        for ref, style in (("full", "--"), ("none", ":")):
            v = [t["correct"] for t in rows if t["method"] == ref]
            if v:
                ax.axhline(v[0], ls=style, color="#111827", lw=1, label=f"{ref} context")
        ax.set_title(model)
        ax.set_xlabel("fraction of tokens kept")
        ax.set_ylabel("answers correct")
        ax.invert_xaxis()
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_compression.png", dpi=200)
    print("Saved results/accuracy_vs_compression.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
