"""Ijaz demo: watch Arabic context compression step by step, and ask GPT / Claude live.

Run: streamlit run app.py
Every number on screen comes from the models or from the saved results in results/.
"""
import html
import json
import os
import time
from math import comb

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from ijaz.learned import plan_final, word_probs
from ijaz.tokens import count_tokens

load_dotenv()
st.set_page_config(page_title="Ijaz · إيجاز", page_icon="✂️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600&display=swap');
.hero h1 {font-size: 2.6rem; margin-bottom: 0;}
.hero p {opacity: .75; font-size: 1.1rem; margin-top: .2rem;}
.cards {display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 18px 0 8px;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 16px 18px; background: rgba(128,128,128,.06);}
.card .big {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.card .big small {font-size: 1rem; font-weight: 500; opacity: .6;}
.card .lbl {opacity: .7; font-size: .9rem; margin-top: 6px;}
.passage {direction: rtl; text-align: right; font-family: 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif;
          font-size: 1.15rem; line-height: 2.5; padding: 14px 16px; border-radius: 12px;
          border: 1px solid rgba(128,128,128,.2);}
.w {display: inline-block; padding: 0 5px; margin: 1px 1px; border-radius: 6px; transition: all .15s;}
.del {opacity: .22; text-decoration: line-through;}
.cut {background: rgba(239,68,68,.55) !important; opacity: 1;}
.prot {box-shadow: inset 0 0 0 2px #16a34a;}
.glue {border-bottom: 2px dotted rgba(128,128,128,.8);}
.legend {display: flex; flex-wrap: wrap; gap: 14px; font-size: .88rem; opacity: .85; margin: 6px 0 2px;}
.legend span.sw {display: inline-block; width: 14px; height: 14px; border-radius: 4px; margin-right: 6px; vertical-align: -2px;}
.bar {height: 10px; border-radius: 6px; background: rgba(128,128,128,.18); overflow: hidden; margin: 4px 0 10px;}
.bar div {height: 100%; background: linear-gradient(90deg, #14b8a6, #0ea5e9);}
.ans {direction: rtl; text-align: right; font-family: 'IBM Plex Sans Arabic', sans-serif; font-size: 1.05rem;
      border-radius: 10px; padding: 10px 12px; border: 1px solid rgba(128,128,128,.25); min-height: 64px;}
.ok {border-color: #16a34a; background: rgba(22,163,74,.08);}
.bad {border-color: #dc2626; background: rgba(220,38,38,.08);}
.meta {font-size: .82rem; opacity: .7; margin-bottom: 4px;}
</style>""", unsafe_allow_html=True)

RATES = [0.75, 0.5, 0.33]
LEVELS = [("Keep · المتن", "rgba(59,130,246,.32)"), ("Recoverable · اختصار", "rgba(239,68,68,.32)"),
          ("Secondary · إسناد/شاهد", "rgba(245,158,11,.38)"), ("Complement · فضلة", "rgba(168,85,247,.32)")]


# ---------------------------------------------------------------- data from results/
def score_answer(pred, golds):
    from scripts.summarize import score
    return bool(score({"answers": golds, "prediction": pred or ""})[2])


def sign_p(a, b):
    n, k = a + b, max(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n) if n else 1.0


@st.cache_data
def load_saved():
    """Test questions with saved answers for full / ijaz_final / llmlingua2."""
    rows = [json.loads(l) for l in open("results/raw.jsonl", encoding="utf-8")]
    ctx = {}
    for l in open("results/compressions.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["method"] == "full":
            ctx[r["id"]] = r["context"]
    ex = {}
    for r in rows:
        if r.get("extra") or r["method"] not in ("full", "ijaz_final", "llmlingua2") or r["id"] not in ctx:
            continue
        e = ex.setdefault(r["id"], {"id": r["id"], "question": r["question"], "answers": r["answers"],
                                    "context": ctx[r["id"]], "pred": {}})
        e["pred"][(r["method"], r["rate"], r["provider"])] = (r["prediction"], r["input_tokens"],
                                                             score_answer(r["prediction"], r["answers"]))
    return [e for e in ex.values() if any(k[0] == "ijaz_final" for k in e["pred"])]


@st.cache_data
def load_results():
    s = pd.read_csv("results/summary.csv")
    s = s[s["method"].isin(["full", "none", "llmlingua2", "ijaz_final"])]
    ex = load_saved()
    paired = []
    for rate in RATES:
        a = sum(e["pred"].get(("ijaz_final", rate, p), (0, 0, False))[2] and not e["pred"].get(("llmlingua2", rate, p), (0, 0, False))[2]
                for e in ex for p in ("openai", "anthropic"))
        b = sum(e["pred"].get(("llmlingua2", rate, p), (0, 0, False))[2] and not e["pred"].get(("ijaz_final", rate, p), (0, 0, False))[2]
                for e in ex for p in ("openai", "anthropic"))
        paired.append({"keep": rate, "only Ijaz right": a, "only LLMLingua-2 right": b, "p": sign_p(a, b)})
    tax = pd.read_csv("results/token_tax.csv").set_index("tokenizer")["arabic_premium"]
    return s, pd.DataFrame(paired), tax


# ---------------------------------------------------------------- compression (cached)
@st.cache_resource(show_spinner="Loading Ijaz models (once)…")
def warm_up():
    """Load both CAMeLBERT models and CAMeL Tools once, so the timer shows compression only."""
    for r in (0.75, 0.33):
        plan_final("ذهب الطالب إلى المدرسة في الصباح الباكر ولم يتأخر عن موعده.", r)
    return True


@st.cache_data(show_spinner=False)
def run_ijaz(text, rate):
    warm_up()
    t0 = time.perf_counter()
    p = plan_final(text, rate)
    p["seconds"] = time.perf_counter() - t0
    p["protected"] = sorted(p["protected"])
    if p["teacher"] is None and p["words"]:  # light mode skips the teacher; still show its view
        p["teacher_view"] = word_probs(p["words"], model_dir="models/ijaz-student-5k")
        p["teacher_used"] = False
    else:
        p["teacher_view"], p["teacher_used"] = p["teacher"], True
    return p


@st.cache_data(show_spinner=False)
def run_ll2(text, rate):
    from ijaz.baselines import llmlingua2
    return llmlingua2(text, rate)


def ll2_kept(words, out):
    """Which original words LLMLingua-2 kept (it keeps word order)."""
    import re
    bare = lambda w: re.sub(r"[^\w]", "", w)  # noqa: E731
    ow, j, mask = out.split(), 0, []
    for w in words:
        if j < len(ow) and (ow[j] == w or bare(ow[j]) == bare(w)):
            mask.append(True)
            j += 1
        else:
            mask.append(False)
    return mask


# ---------------------------------------------------------------- rendering
def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    for pos, i in enumerate(order):
        r[i] = pos / max(1, len(vals) - 1)
    return r


def chips(words, bg=None, cls=None, tips=None):
    out = []
    for i, w in enumerate(words):
        style = f' style="background:{bg[i]}"' if bg and bg[i] else ""
        c = "w " + (cls[i] if cls else "")
        tip = f' title="{html.escape(tips[i])}"' if tips else ""
        out.append(f'<span class="{c}"{style}{tip}>{html.escape(w)}</span>')
    return '<div class="passage">' + " ".join(out) + "</div>"


def legend(items):
    return '<div class="legend">' + "".join(f'<span><span class="sw" style="background:{c}"></span>{html.escape(t)}</span>'
                                            for t, c in items) + "</div>"


def bar(frac, label):
    return f'<div class="meta">{label}</div><div class="bar"><div style="width:{100 * frac:.1f}%"></div></div>'


def show_process(text, rate, key):
    with st.spinner("Ijaz is reading the passage…"):
        p = run_ijaz(text, rate)
    words, n = p["words"], len(p["words"])
    prot = set(p["protected"])
    teacher = p["teacher_view"]
    lv = [max(range(4), key=lambda k: q[k]) for q in teacher]
    ans = p["answerness"] or [0.0] * n
    tips = [f"keep score {s:.2f} · teacher: {LEVELS[l][0]} ({max(q):.0%}) · answerness {a:.2f}"
            + (" · protected" if i in prot else "") for i, (s, l, q, a) in enumerate(zip(p["score"], lv, teacher, ans))]

    st.markdown("#### How Ijaz decides")
    st.markdown(f'<div class="card" style="margin:6px 0 10px"><div class="big">Keep {int(rate * 100)}% '
                f'<small>→ send {p["budget"]} of {p["original_tokens"]} tokens · {len(p["deleted"])} of {n} words will be deleted'
                f'</small></div></div>', unsafe_allow_html=True)
    mode = ("Light compression: Ijaz ranks words by answerness (the teacher step is skipped; on tuning questions it hurt here)."
            if not p["teacher_used"] else
            "Heavy compression: Ijaz combines the teacher (what can be dropped safely) with answerness (what is worth keeping).")
    st.caption(mode + "  Steps ②–④ score the passage itself, so they look the same at any keep rate; "
               "the keep rate only decides where the cut falls in step ⑤.")
    s1, s2, s3, s4, s5 = st.tabs(["① Read", "② Teacher · الحذف", "③ Answerness", "④ Combine & protect", "⑤ Delete to budget"])
    with s1:
        c = st.columns(4)
        c[0].metric("Words", n)
        c[1].metric("Tokens (GPT o200k)", p["original_tokens"])
        c[2].metric("Budget", p["budget"], f"keep {int(rate * 100)}%", delta_color="off")
        c[3].metric("Ijaz time", f"{p['seconds'] * 1000:.0f} ms" if p["seconds"] > 0.001 else "cached")
        st.markdown(chips(words), unsafe_allow_html=True)
        st.caption("The passage is split into words. The budget is the number of tokens the LLM will be sent.")
    with s2:
        st.markdown(legend(LEVELS), unsafe_allow_html=True)
        st.markdown(chips(words, bg=[LEVELS[l][1] for l in lv], tips=tips), unsafe_allow_html=True)
        st.caption("A CAMeLBERT student (110M) trained on 5,000 passages labelled by a GPT teacher that applied the classical "
                   "rules of الحذف: drop first what the reader can recover (اختصار), then secondary information, then فضلة."
                   + ("" if p["teacher_used"] else "  ·  Shown for reference: not used at this keep rate."))
    with s3:
        r = ans
        st.markdown(legend([("unlikely to be asked about", "rgba(20,184,166,.06)"), ("likely part of an answer", "rgba(20,184,166,.75)")]),
                    unsafe_allow_html=True)
        st.markdown(chips(words, bg=[f"rgba(20,184,166,{0.05 + 0.7 * a:.2f})" for a in r], tips=tips), unsafe_allow_html=True)
        st.caption("A second CAMeLBERT trained on 10,318 Arabic Wikipedia passages where people wrote real questions and marked "
                   "the answers (TyDiQA). It learned which words carry facts people ask about. No question is needed at compression time.")
    with s4:
        rk = ranks(p["score"])
        st.markdown(legend([("delete first", "rgba(14,165,233,.06)"), ("keep longest", "rgba(14,165,233,.75)"),
                            ("protected: never deleted", "transparent")]) +
                    '<div class="legend"><span><span class="w prot">١٩٢٣</span> numbers, negation (لم، لا، ليس، غير…), '
                    'exception, condition, contrast, and the word after each</span>'
                    '<span><span class="w glue">في</span> glue words, lowered: an LLM restores them itself</span></div>',
                    unsafe_allow_html=True)
        st.markdown(chips(words, bg=[f"rgba(14,165,233,{0.05 + 0.7 * x:.2f})" for x in rk],
                          cls=[("prot " if i in prot else "") + ("glue" if p["glue"][i] else "") for i in range(n)], tips=tips),
                    unsafe_allow_html=True)
        st.caption("One keep score per word. Hard rules from Ibn Hisham's conditions of الحذف (أمن اللبس، ألا يكون كالجزء) "
                   "protect words whose loss would flip or break the meaning.")
    with s5:
        holder = st.empty()
        deleted = p["deleted"]

        def frame(k, flash=()):
            gone = set(deleted[:k])
            kept_text = " ".join(w for i, w in enumerate(words) if i not in gone)
            tok = count_tokens(kept_text)
            cls = [("del" if i in gone and i not in flash else "") + (" cut" if i in flash else "") + (" prot" if i in prot else "")
                   for i in range(n)]
            holder.markdown(bar(tok / p["original_tokens"], f"{tok} / {p['original_tokens']} tokens  ·  "
                                                            f"{k} of {len(deleted)} words deleted  ·  target {p['budget']}")
                            + chips(words, cls=cls, tips=tips), unsafe_allow_html=True)

        if st.button("▶ Play the deletion", key=f"play-{key}"):
            step = max(1, len(deleted) // 45)
            for k in range(0, len(deleted), step):
                frame(k, flash=set(deleted[k:k + step]))
                time.sleep(0.09)
        frame(len(deleted))
        st.caption("Words go in order of their keep score, lowest first, until the text fits the budget.")
    return p


def show_side_by_side(text, rate, p):
    with st.spinner("LLMLingua-2 (Microsoft) is compressing the same passage…"):
        ll = run_ll2(text, rate)
    words = p["words"]
    keep_ll = ll2_kept(words, ll)
    st.markdown("#### Ijaz vs LLMLingua-2 at the same budget")
    a, b = st.columns(2)
    with a:
        st.markdown(f"**Ijaz** · {p['final_tokens']} tokens")
        st.markdown(bar(p["final_tokens"] / p["original_tokens"], "of the original"), unsafe_allow_html=True)
        st.markdown(chips(words, cls=["" if k else "del" for k in p["keep"]]), unsafe_allow_html=True)
    with b:
        t = count_tokens(ll)
        st.markdown(f"**LLMLingua-2** (Microsoft, 560M) · {t} tokens")
        st.markdown(bar(t / p["original_tokens"], "of the original"), unsafe_allow_html=True)
        st.markdown(chips(words, cls=["" if k else "del" for k in keep_ll]), unsafe_allow_html=True)
    return {"Full text": text, "Ijaz": p["text"], "LLMLingua-2": ll}


def answer_grid(results, golds):
    """results: {provider: {variant: (answer, input_tokens)}}"""
    names = {"openai": os.getenv("OPENAI_MODEL", "gpt-5.4-mini"), "anthropic": os.getenv("CLAUDE_MODEL", "claude-haiku-4-5")}
    for prov, res in results.items():
        st.markdown(f"**{names.get(prov, prov)}**")
        cols = st.columns(len(res))
        for col, (variant, (ans, tok)) in zip(cols, res.items()):
            ok = score_answer(ans, golds) if golds else None
            mark = "" if ok is None else (" ✓" if ok else " ✗")
            col.markdown(f'<div class="meta">{variant} · {tok} input tokens{mark}</div>'
                         f'<div class="ans {"" if ok is None else ("ok" if ok else "bad")}">{html.escape(ans or "(no answer)")}</div>',
                         unsafe_allow_html=True)


# ---------------------------------------------------------------- page
summary, paired, tax = load_results()


def pct(model, method, rate):
    v = summary[(summary.model.str.startswith(model)) & (summary.method == method) & (summary.keep_rate == rate)]["correct"]
    return f"{100 * v.iloc[0]:.1f}%" if len(v) else "–"


p33 = paired[paired.keep == 0.33].iloc[0]
st.markdown('<div class="hero"><h1>إيجاز · Ijaz</h1><p>Arabic context compression for any LLM. '
            'Fewer tokens in, the facts stay. Question-free, runs locally.</p></div>', unsafe_allow_html=True)
st.markdown(f"""<div class="cards">
<div class="card"><div class="big">{pct('gpt', 'ijaz_final', 0.33)} <small>vs {pct('gpt', 'llmlingua2', 0.33)}</small></div>
<div class="lbl">GPT-5.4-mini correct answers at ⅓ of the tokens: Ijaz vs LLMLingua-2</div></div>
<div class="card"><div class="big">{pct('claude', 'ijaz_final', 0.33)} <small>vs {pct('claude', 'llmlingua2', 0.33)}</small></div>
<div class="lbl">Claude Haiku 4.5, same test</div></div>
<div class="card"><div class="big">{int(p33['only Ijaz right'])} <small>vs {int(p33['only LLMLingua-2 right'])}</small></div>
<div class="lbl">questions only one of them got right at ⅓ (p = {p33['p']:.3f}, 200 native Arabic questions × 2 models)</div></div>
<div class="card"><div class="big">{tax.get('Claude', float('nan')):.2f}× <small>Claude · {tax.get('GPT o200k_base', float('nan')):.2f}× GPT</small></div>
<div class="lbl">Arabic token tax: tokens for the same content vs English (ALLaM: {tax.get('ALLaM-7B', float('nan')):.2f}×)</div></div>
</div>""", unsafe_allow_html=True)

live, saved, results = st.tabs(["⚡ Live demo", "📚 Test examples", "📊 Results"])

with live:
    probe = json.load(open("data/negation_probe.json", encoding="utf-8"))
    st.markdown("Paste any Arabic passage (MSA), pick how much to keep, and watch Ijaz work.")
    c1, c2 = st.columns([3, 1])
    with c1:
        text = st.text_area("Arabic passage", value=probe[0]["context"], height=150, key="live_text")
    with c2:
        rate = st.segmented_control("Keep", RATES, default=0.33, format_func=lambda r: f"{int(r * 100)}%", key="live_rate") or 0.33
    if text.strip():
        p = show_process(text, rate, "live")
        show_side_by_side(text, rate, p)

with saved:
    ex = load_saved()
    st.markdown("Real questions from the test set (ARCD, native Arabic Wikipedia). Answers below are the **saved** answers "
                "from the evaluation run, not regenerated.")
    c1, c2 = st.columns([1, 3])
    with c1:
        srate = st.segmented_control("Keep", RATES, default=0.33, format_func=lambda r: f"{int(r * 100)}%", key="saved_rate") or 0.33
        show = st.radio("Show", ["All", "Ijaz right, LLMLingua-2 wrong", "LLMLingua-2 right, Ijaz wrong"], key="saved_filter")

    def right(e, m):
        return sum(e["pred"].get((m, srate, pv), (0, 0, False))[2] for pv in ("openai", "anthropic"))

    pool = ex if show == "All" else [e for e in ex if (right(e, "ijaz_final") > right(e, "llmlingua2")) == show.startswith("Ijaz")
                                    and right(e, "ijaz_final") != right(e, "llmlingua2")]
    with c2:
        pick = st.selectbox(f"Question ({len(pool)} shown)", range(len(pool)), format_func=lambda i: pool[i]["question"],
                            key=f"saved_pick-{show}-{srate}")
    if pool:
        e = pool[pick]
        st.markdown(f'<div class="meta">Expected answer</div><div class="ans">{html.escape(" / ".join(e["answers"]))}</div>',
                    unsafe_allow_html=True)
        p = show_process(e["context"], srate, "saved")
        show_side_by_side(e["context"], srate, p)
        st.markdown("#### What the models answered (saved evaluation)")
        res = {}
        for pv in ("openai", "anthropic"):
            for name, m, r in (("Full text", "full", 1.0), ("Ijaz", "ijaz_final", srate), ("LLMLingua-2", "llmlingua2", srate)):
                if (m, r, pv) in e["pred"]:
                    res.setdefault(pv, {})[name] = e["pred"][(m, r, pv)][:2]
        answer_grid(res, e["answers"])
        st.caption("✓/✗ uses the standard reading-comprehension rule (word-overlap F1 ≥ 0.5, or the answer contains the "
                   "expected one). It can mark a longer but correct answer wrong, e.g. extra words around the right name.")

with results:
    st.markdown("200 random questions from ARCD (native Arabic), answered by GPT-5.4-mini and Claude Haiku 4.5 "
                "from the full passage, from nothing, and from each compressed version.")
    s = summary.copy()
    s["model"] = s["model"].map(lambda m: "GPT-5.4-mini" if m.startswith("gpt") else "Claude Haiku 4.5")
    s["method"] = s["method"].map({"ijaz_final": "Ijaz", "llmlingua2": "LLMLingua-2", "full": "Full text", "none": "No text"})
    for col, model in zip(st.columns(2), ["GPT-5.4-mini", "Claude Haiku 4.5"]):
        lines = s[(s.model == model) & s.method.isin(["Ijaz", "LLMLingua-2"])]
        refs = s[(s.model == model) & s.method.isin(["Full text", "No text"])]
        base = alt.Chart(lines).encode(
            x=alt.X("keep_rate:Q", title="Share of tokens kept", scale=alt.Scale(domain=[0.3, 0.78]), axis=alt.Axis(format="%")),
            y=alt.Y("correct:Q", title="Correct answers", scale=alt.Scale(domain=[0.2, 0.95]), axis=alt.Axis(format="%")),
            color=alt.Color("method:N", scale=alt.Scale(domain=["Ijaz", "LLMLingua-2"], range=["#14b8a6", "#94a3b8"]),
                            title=None, legend=alt.Legend(orient="bottom")))
        chart = (base.mark_line(strokeWidth=3) + base.mark_circle(size=90)
                 + alt.Chart(refs).mark_rule(strokeDash=[5, 4], opacity=.6).encode(y="correct:Q", color=alt.value("#64748b"))
                 + alt.Chart(refs).mark_text(align="left", dx=4, dy=-7, opacity=.7).encode(
                     x=alt.value(4), y="correct:Q", text="method:N"))
        col.markdown(f"**{model}**")
        col.altair_chart(chart.properties(height=320), width="stretch")
    st.markdown("**Question by question** (both models pooled): how often only one method got it right")
    show_p = paired.copy()
    show_p["keep"] = show_p["keep"].map(lambda r: f"{int(r * 100)}%")
    show_p["verdict"] = [("Ijaz wins" if a > b else "LLMLingua-2 wins") if p < 0.05 else "tie (not significant)"
                         for a, b, p in zip(paired["only Ijaz right"], paired["only LLMLingua-2 right"], paired["p"])]
    show_p["p"] = show_p["p"].map(lambda x: f"{x:.3f}")
    st.dataframe(show_p, hide_index=True, width="stretch")
    st.caption("Scoring: an answer counts if its word-overlap F1 with the expected answer is ≥ 0.5 or it contains it "
               "(same rule as the 'Lost in Compression' audit). Robustness check with a lenient rule that ignores extra words: "
               "at 33% Ijaz still wins (71 vs 39, p = 0.003); at 50% still a tie; at 75% LLMLingua-2 wins (38 vs 19, p = 0.016).  ·  "
               "Honest limits: at 75% LLMLingua-2 is as good or better, and on answers next to a negation it did better at 75%. "
               "Ijaz uses two CAMeLBERT models (2 × 110M parameters) vs LLMLingua-2's 560M.")
