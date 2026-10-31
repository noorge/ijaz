"""How much more does Arabic cost than English for the SAME content?

Uses Belebele (parallel Arabic/English passages) when available, else a small built-in set.
Tokenizers: GPT (o200k, cl100k), Qwen2.5, ALLaM, and Claude via the count_tokens API.

Run:  python scripts/token_tax.py
Output: results/token_tax.csv and a printed table for your opening slide.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

BUILTIN = [
    ("ذهب الطالب إلى المكتبة لاستعارة كتاب عن تاريخ المملكة العربية السعودية.",
     "The student went to the library to borrow a book about the history of Saudi Arabia."),
    ("لم تتمكن الشركة من تحقيق أرباح خلال العام الماضي بسبب ارتفاع تكاليف التشغيل.",
     "The company was unable to make a profit last year due to rising operating costs."),
    ("يعد التعليم من أهم العوامل التي تساهم في تنمية المجتمعات وتطورها.",
     "Education is one of the most important factors that contribute to the development of societies."),
    ("أعلنت وزارة الصحة عن افتتاح ثلاثة مستشفيات جديدة في المنطقة الشرقية.",
     "The Ministry of Health announced the opening of three new hospitals in the Eastern Province."),
    ("يستخدم الذكاء الاصطناعي اليوم في مجالات كثيرة مثل الطب والتعليم والنقل.",
     "Artificial intelligence is used today in many fields such as medicine, education and transport."),
    ("وصل الوفد إلى الرياض صباح يوم الاثنين لحضور المؤتمر الدولي للطاقة.",
     "The delegation arrived in Riyadh on Monday morning to attend the international energy conference."),
    ("إذا لم يتم دفع الفاتورة قبل نهاية الشهر، فسيتم إيقاف الخدمة تلقائيا.",
     "If the bill is not paid before the end of the month, the service will be stopped automatically."),
    ("تقع مدينة العلا في شمال غرب المملكة وتشتهر بمواقعها الأثرية.",
     "The city of AlUla is located in the northwest of the Kingdom and is famous for its archaeological sites."),
]


def load_pairs(limit=300):
    try:
        from datasets import load_dataset
        ar = load_dataset("facebook/belebele", "arb_Arab", split="test")
        en = load_dataset("facebook/belebele", "eng_Latn", split="test")
        seen, pairs = set(), []
        for a, e in zip(ar, en):
            if a["link"] in seen:
                continue
            seen.add(a["link"])
            pairs.append((a["flores_passage"], e["flores_passage"]))
            if len(pairs) >= limit:
                break
        print(f"Using Belebele: {len(pairs)} parallel passages")
        return pairs, "belebele"
    except Exception as ex:
        print(f"Belebele unavailable ({ex.__class__.__name__}); using built-in pairs")
        return BUILTIN, "builtin"


def tokenizers():
    out = {}
    try:
        import tiktoken
        for name in ("o200k_base", "cl100k_base"):
            enc = tiktoken.get_encoding(name)
            out[f"GPT {name}"] = lambda t, e=enc: len(e.encode(t))
    except Exception as ex:
        print("tiktoken failed:", ex)
    for label, repo in (("Qwen2.5", "Qwen/Qwen2.5-7B-Instruct"),
                        ("ALLaM-7B", "ALLaM-AI/ALLaM-7B-Instruct-preview")):
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(repo)
            out[label] = lambda t, k=tok: len(k.encode(t, add_special_tokens=False))
        except Exception as ex:
            print(f"{label} tokenizer failed: {ex.__class__.__name__}")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")
            base = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": "x"}]).input_tokens

            def claude_count(t):
                r = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": t}])
                return r.input_tokens - base + 1
            out["Claude"] = claude_count
        except Exception as ex:
            print("Claude count failed:", ex)
    return out


def main():
    pairs, source = load_pairs()
    toks = tokenizers()
    claude_limit = 40  # count_tokens is an API call; a sample is enough
    rows = []
    for name, fn in toks.items():
        sample = pairs[:claude_limit] if name == "Claude" else pairs
        ar = sum(fn(a) for a, _ in sample)
        en = sum(fn(e) for _, e in sample)
        rows.append({"tokenizer": name, "arabic_tokens": ar, "english_tokens": en,
                     "arabic_premium": round(ar / en, 2), "passages": len(sample), "source": source})
    os.makedirs("results", exist_ok=True)
    with open("results/token_tax.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'Tokenizer':<18}{'Arabic':>10}{'English':>10}{'Arabic costs':>16}")
    for r in rows:
        print(f"{r['tokenizer']:<18}{r['arabic_tokens']:>10}{r['english_tokens']:>10}{r['arabic_premium']:>14}x")
    print("\nSaved results/token_tax.csv")


if __name__ == "__main__":
    main()
