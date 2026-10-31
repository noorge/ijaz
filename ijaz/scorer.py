"""Layer 3+4: score how much each word matters, and protect what must never go.

Score scale: 0 = drop first, 1 = very important, inf = protected.
Rule score (always) can be blended with a small language model's surprisal (optional).
"""
import math
import re
from collections import Counter

from .lexicon import PROTECT, FUNCTION_WEIGHTS, NEGATIONS
from .morph import split_clitics

PROTECTED = math.inf
_DIGIT = re.compile(r"[0-9٠-٩۰-۹]")
_LATIN = re.compile(r"[A-Za-z]")
_ARABIC = re.compile(r"[؀-ۿ]")
_HARD_PUNCT = {".", "؟", "?", "!", ":", "\n", "«", "»", '"', "(", ")"}
# Words that announce a name right after them
_TITLES = {"الملك", "الأمير", "الشيخ", "الدكتور", "الرئيس", "الإمام", "السلطان", "مدينة", "جامعة",
           "نهر", "جبل", "دولة", "شركة", "مملكة", "جمهورية", "بحر", "منطقة", "محافظة"}
_NAME_LINKS = {"بن", "بنت", "ابن", "أبو", "ابو", "أم", "آل"}
_POS_WEIGHT = {"verb": 0.65, "noun": 0.7, "adj": 0.45, "adv": 0.5, "noun_num": PROTECTED,
               "digit": PROTECTED, "noun_prop": PROTECTED, "part_neg": PROTECTED, "abbrev": 0.8}


def _strip_conj(word):
    return word[1:] if len(word) > 2 and word[0] in "وف" else word


def _is_protected_word(w):
    if w in PROTECT or _strip_conj(w) in PROTECT:
        return True
    pro, stem, enc = split_clitics(w)
    return stem in PROTECT or ("ال" + stem) in PROTECT


def rule_scores(tokens, pos=None):
    """tokens: list of words/punctuation. pos: optional CAMeL POS tag per token."""
    stems = [split_clitics(t)[1] if _ARABIC.search(t) else t for t in tokens]
    stem_freq = Counter(s for s in stems if len(s) >= 3)
    seen = set()
    scores = []
    for i, tok in enumerate(tokens):
        # punctuation and structure
        if not re.search(r"\w", tok):
            between_numbers = (tok in "-–/" and 0 < i < len(tokens) - 1
                               and _DIGIT.search(tokens[i - 1]) and _DIGIT.search(tokens[i + 1]))
            scores.append(PROTECTED if tok in _HARD_PUNCT or between_numbers else 0.3)
            continue
        if tok.startswith("«") or _DIGIT.search(tok) or _LATIN.search(tok):
            scores.append(PROTECTED)  # numbers, Latin names/acronyms, dictionary aliases
            continue
        if _is_protected_word(tok):
            scores.append(PROTECTED)
            continue
        bare = _strip_conj(tok)
        prev = tokens[i - 1] if i > 0 else ""
        if prev in NEGATIONS or _strip_conj(prev) in NEGATIONS:
            scores.append(0.97)  # whatever a negation applies to stays, even "تكون"
            continue
        if tok in FUNCTION_WEIGHTS or bare in FUNCTION_WEIGHTS:
            scores.append(FUNCTION_WEIGHTS.get(tok, FUNCTION_WEIGHTS.get(bare)))
            continue
        tag = pos[i] if pos else None
        if tok in _NAME_LINKS:  # بن / أبو inside a name chain
            scores.append(0.9)
            continue
        if tag in _POS_WEIGHT:
            s = _POS_WEIGHT[tag]
        elif tag and tag.startswith("prep"):
            s = 0.4
        else:
            s = 0.6
        stem = stems[i]
        if s == PROTECTED:
            if tag == "noun_prop" and stem in seen and tokens[i - 1] not in {"بن", "بنت", "ابن", "عبد"}:
                s = 0.55  # a name already mentioned: later mentions can go
            else:
                seen.add(stem)
                scores.append(s)
                continue
        # name heuristics (matter most when CAMeL is not installed)
        if i > 0 and tokens[i - 1] in _TITLES:
            s = max(s, 0.95)
        if pos is None and stem_freq.get(stem, 0) >= 2 and stem not in seen:
            s = max(s, 0.85)  # recurring term = likely topic entity; keep first mention
        # redundancy: a repeated mention carries less new information (the zip idea)
        if stem in seen:
            s *= 0.7
        s += min(0.15, max(0, len(stem) - 4) * 0.04)  # longer stems tend to be content
        prev = tokens[i - 1] if i > 0 else ""
        # negation scope: "لم [تحقق]" means nothing without the verb it negates
        if prev in NEGATIONS or _strip_conj(prev) in NEGATIONS:
            s = max(s, 0.97)
        # anchor scope: "بسبب [المرض]", "يوم [الأحد]", "قبل [الحرب]" — the anchor is empty alone
        elif prev in PROTECT or _strip_conj(prev) in PROTECT:
            s = max(s, 0.9)
        # Arabic is verb-subject-object: a noun right after a verb is usually who did it
        if pos and i > 0 and pos[i - 1] == "verb" and tag in ("noun", "noun_prop"):
            s = max(s, 0.85)
        # inside a name chain: بن [عبد], عبد [المطلب]
        if prev in _NAME_LINKS or prev == "عبد":
            s = max(s, 0.9)
        seen.add(stem)
        scores.append(min(s, 0.99))
    return scores


def lm_surprisal(text, tokens, model_name="Qwen/Qwen2.5-0.5B", device=None):
    """Optional: per-word surprisal from a small causal LM, scaled to 0-1 by rank.

    Predictable words (low surprisal) are cheaper to drop. Runs on a laptop GPU.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _LM
    if "_LM" not in globals() or _LM[0] != model_name:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        _LM = (model_name, tok, mdl.to(dev).eval(), dev)
    _, tok, mdl, dev = _LM

    # rebuild text from tokens so character offsets line up
    spans, pos, joined = [], 0, []
    for t in tokens:
        if joined:
            joined.append(" ")
            pos += 1
        spans.append((pos, pos + len(t)))
        joined.append(t)
        pos += len(t)
    full = "".join(joined)
    enc = tok(full, return_offsets_mapping=True, return_tensors="pt")
    ids = enc["input_ids"].to(dev)
    with torch.no_grad():
        logits = mdl(ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1)
    nll = -logp.gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1).cpu().tolist()
    offsets = enc["offset_mapping"][0].tolist()[1:]
    word_s = [0.0] * len(tokens)
    for (a, b), v in zip(offsets, nll):
        for wi, (s, e) in enumerate(spans):
            if a < e and b > s:
                word_s[wi] += v
                break
    order = sorted(range(len(tokens)), key=lambda k: word_s[k])
    ranks = [0.0] * len(tokens)
    for r, k in enumerate(order):
        ranks[k] = r / max(1, len(tokens) - 1)
    return ranks
