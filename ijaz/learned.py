"""Learned Ijaz: a small Arabic BERT (CAMeLBERT-MSA, 110M) trained on teacher labels.

The teacher applied the classical rules of الحذف in data/teacher_prompt.txt; the student learned to
predict, for every word, its deletion level:
  0 keep (المتن) | 1 اختصار: recoverable, drop first | 2 secondary info | 3 فضلة: drop last
To hit a token budget we drop words from level 1 upward. Digits and negation/exception/condition/
contrast words (plus the word right after each) are never dropped, same as the teacher labels.
"""
import math
import re

from .lexicon import NEGATIONS
from .tokens import count_tokens, word_cost

MODEL_DIR = "models/ijaz-student"
LEVEL_VALUE = [1.0, 0.0, 1 / 3, 2 / 3]  # how much each class is worth keeping: 0 keep ... 1 drop first
NEVER = NEGATIONS | {"إلا", "الا", "سوى", "عدا", "لكن", "ولكن", "لكنه", "لكنها", "بل", "إن", "إذا", "اذا", "لو"}
_DIGIT = re.compile(r"[0-9٠-٩]")

_models = {}  # model_dir -> (tokenizer, model, device), so several students can be compared


def never_drop(word):
    return re.sub(r"[^\w]", "", word) in NEVER or bool(_DIGIT.search(word))


def protected_indices(words):
    """Never-drop words plus the word right after each (its scope: لم [تتطور], 27 [محافظة])."""
    return {j for i, w in enumerate(words) if never_drop(w) for j in (i, i + 1)}


def _load(model_dir=MODEL_DIR):
    if model_dir not in _models:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(model_dir)
        mdl = AutoModelForTokenClassification.from_pretrained(model_dir).to(dev).eval()
        _models[model_dir] = (tok, mdl, dev)
    return _models[model_dir]


def word_probs(words, window=200, model_dir=MODEL_DIR):
    """Per word: probabilities of levels 0-3. Read in 200-word windows (BERT takes 512 subwords)."""
    import torch
    tok, mdl, dev = _load(model_dir)
    out = []
    for s in range(0, len(words), window):
        chunk = words[s:s + window]
        enc = tok(chunk, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            p = mdl(**{k: v.to(dev) for k, v in enc.items()}).logits[0].softmax(-1).cpu().tolist()
        first = {}
        for t, w in enumerate(enc.word_ids()):
            if w is not None and w not in first:
                first[w] = p[t]
        out += [first.get(i, [1.0, 0.0, 0.0, 0.0]) for i in range(len(chunk))]
    return out


ANS_DIR = "models/ijaz-answerness"  # P(word is part of an answer), trained on TyDiQA Arabic
GLUE = ("prep", "conj", "pron", "part")  # an LLM reader restores these itself
_OTHER_SCRIPT = re.compile(r"[A-Za-z֐-׿]")


def plan(text, keep_rate=0.5, model_dir=MODEL_DIR, ans_weight=0.0, teacher=True, glue=0.0, soft=0.0):
    """Every decision behind one compression, step by step (the demo shows each step).

    ans_weight: add ans_weight * answerness to each word's keep score.
    teacher=False: rank by answerness alone. glue: lower the keep score of glue words (CAMeL POS).
    soft: raise the keep score of Latin/Hebrew words and «quoted» words.
    """
    words = text.split()
    n = len(words)
    p = {"words": words, "teacher": None, "answerness": None, "glue": [False] * n, "soft": [False] * n}
    if teacher:
        p["teacher"] = word_probs(words, model_dir=model_dir) if words else []
        score = [sum(q * v for q, v in zip(pr, LEVEL_VALUE)) for pr in p["teacher"]]
    else:
        score = [0.0] * n
    if words and (ans_weight or not teacher):
        p["answerness"] = [q[1] for q in word_probs(words, model_dir=ANS_DIR)]
        score = [s + (ans_weight if teacher else 1.0) * a for s, a in zip(score, p["answerness"])]
    if words and glue:
        from .morph import pos_tags
        tags = pos_tags([re.sub(r"[^\w]", "", w) or w for w in words]) or []
        p["glue"] = [t.startswith(GLUE) for t in tags] + [False] * (n - len(tags))
        score = [s - glue if g else s for s, g in zip(score, p["glue"])]
    if soft:
        p["soft"] = [bool(_OTHER_SCRIPT.search(w)) or "«" in w or "»" in w for w in words]
        score = [s + soft if f else s for s, f in zip(score, p["soft"])]
    protected = protected_indices(words)
    order = sorted((i for i in range(n) if i not in protected), key=lambda i: score[i])
    original = count_tokens(text)
    budget = math.floor(keep_rate * original)
    keep = [True] * n
    current = original
    k = 0
    while k < len(order) and current > budget:
        keep[order[k]] = False
        current -= word_cost(words[order[k]])
        k += 1
        if current <= budget:  # estimate says done: measure the real text, keep going if over
            current = count_tokens(" ".join(w for w, kp in zip(words, keep) if kp))
    out = " ".join(w for w, kp in zip(words, keep) if kp) if words else text
    p.update(score=score, protected=protected, deleted=order[:k], keep=keep, text=out,
             original_tokens=original, budget=budget, final_tokens=count_tokens(out))
    return p


def compress_learned(text, keep_rate=0.5, **kw):
    return plan(text, keep_rate, **kw)["text"]


def final_settings(keep_rate):
    """ijaz_final, chosen on dev questions only: the teacher hurt at light compression and helped at heavy."""
    if keep_rate >= 0.7:
        return {"teacher": False, "glue": 0.3, "soft": 0.5}
    return {"model_dir": "models/ijaz-student-5k", "ans_weight": 2.0, "glue": 0.3, "soft": 0.5}


def plan_final(text, keep_rate=0.5):
    return plan(text, keep_rate, **final_settings(keep_rate))


def compress_final(text, keep_rate=0.5):
    return plan_final(text, keep_rate)["text"]
