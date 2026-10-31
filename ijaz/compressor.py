"""Ijaz: Arabic-aware context compression.

compress(text, keep_rate) -> shorter Arabic text, readable by any model.
Pipeline: clean -> concise rewrites -> repeated-phrase dictionary -> analyze words
-> score + protect -> drop lowest scores until the token budget is met -> rebuild.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .lexicon import FILLER_PHRASES, CONJ_CARRIERS
from .morph import pos_tags
from .normalize import normalize
from .scorer import rule_scores, lm_surprisal, PROTECTED
from .tokens import count_tokens, word_cost

_TOKEN_RE = re.compile(r"\n|«\d+»|\w+|[^\w\s]")
_NO_SPACE_BEFORE = set("،؛؟.!:,;?)]»")
_NO_SPACE_AFTER = set("([«")


@dataclass
class CompressionResult:
    text: str
    original_tokens: int
    compressed_tokens: int
    target_rate: float
    dropped: list = field(default_factory=list)
    protected_count: int = 0
    steps: dict = field(default_factory=dict)  # tokens after each layer, for the demo
    view: list = field(default_factory=list)   # (token, "kept"|"dropped"|"protected") for highlighting

    @property
    def achieved_rate(self):
        return self.compressed_tokens / max(1, self.original_tokens)


def _apply_fillers(text):
    for phrase, repl in sorted(FILLER_PHRASES, key=lambda p: -len(p[0])):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        text = re.sub(pattern, repl, text)
    text = re.sub(r"(?<!\w)و\s+", "و", text)  # glue a stranded "و" to the next word
    return re.sub(r" {2,}", " ", text).strip()


def _apply_dictionary(text, min_words=3, min_count=2):
    """Replace long repeated phrases with short aliases «1», «2" and a legend.
    Kept only if it actually saves tokens (lossless, the zip/LZ idea)."""
    words = text.split()
    best = {}
    for n in range(8, min_words - 1, -1):
        grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        for g, c in grams.items():
            if c >= min_count and not any(set(g) <= set(k) for k in best):
                phrase = " ".join(g)
                saving = (c - 1) * count_tokens(phrase) - c * 2 - 4
                if saving > 0:
                    best[g] = (phrase, saving)
    if not best:
        return text, {}
    legend = {}
    for i, (g, (phrase, _)) in enumerate(sorted(best.items(), key=lambda x: -x[1][1])[:9], 1):
        alias = f"«{i}»"
        if phrase in text:
            text = text.replace(phrase, alias)
            legend[alias] = phrase
    return text, legend


def _rebuild(tokens):
    out = ""
    for t in tokens:
        if not out or out.endswith(("\n",)) or out[-1] in _NO_SPACE_AFTER or t in _NO_SPACE_BEFORE or t == "\n":
            out += t
        else:
            out += " " + t
    # tidy orphan punctuation left by deletions
    out = re.sub(r"([،,؛;])\s*(?=[،,؛;.؟?!])", "", out)
    out = re.sub(r"(^|\n)\s*[،,؛;]\s*", r"\1", out)
    out = re.sub(r"\(\s*\)", "", out)
    return re.sub(r" {2,}", " ", out).strip()


def _assemble(tokens, keep, pos):
    """Collect kept tokens. A dropped conjunction (و/ف) is handed to the next kept
    word so sentences stay linked instead of becoming a word list."""
    final, carry = [], ""
    for i, t in enumerate(tokens):
        if not keep[i]:
            if t in CONJ_CARRIERS or (pos and pos[i] == "conj"):
                carry = t[0] if t[0] in "وف" else carry
            continue
        if carry and re.match(r"\w", t) and t[0] not in "وف" and not t.startswith("«"):
            t = carry + t
        carry = ""
        final.append(t)
    return final


def compress(text, keep_rate=0.5, use_lm=False, lm_name="Qwen/Qwen2.5-0.5B",
             keep_diacritics=False, use_dictionary=True, use_morphology=True):
    original_tokens = count_tokens(text)
    budget = math.floor(keep_rate * original_tokens)
    steps = {"original": original_tokens}

    work = normalize(text, keep_diacritics=keep_diacritics)
    steps["clean"] = count_tokens(work)
    work = _apply_fillers(work)
    steps["concise"] = count_tokens(work)

    legend = {}
    if use_dictionary:
        work, legend = _apply_dictionary(work)
    legend_text = "".join(f"{k}={v}\n" for k, v in legend.items())
    steps["dictionary"] = count_tokens(legend_text + work)

    tokens = _TOKEN_RE.findall(work)
    pos = pos_tags(tokens) if use_morphology else None
    scores = rule_scores(tokens, pos)
    if use_lm:
        lm = lm_surprisal(work, tokens, lm_name)
        scores = [s if s == PROTECTED else 0.5 * s + 0.5 * l for s, l in zip(scores, lm)]

    order = sorted((i for i, s in enumerate(scores) if s != PROTECTED), key=lambda i: scores[i])
    keep = [True] * len(tokens)
    current = steps["dictionary"]
    dropped = []
    k = 0
    while k < len(order) and current > budget:
        i = order[k]
        keep[i] = False
        current -= word_cost(tokens[i])
        dropped.append(tokens[i])
        k += 1
        if current <= budget:  # estimate says done: measure the real text, keep going if over
            current = count_tokens(legend_text + _rebuild(_assemble(tokens, keep, pos)))
    out = legend_text + _rebuild(_assemble(tokens, keep, pos))
    return CompressionResult(
        text=out,
        original_tokens=original_tokens,
        compressed_tokens=count_tokens(out),
        target_rate=keep_rate,
        dropped=dropped,
        protected_count=sum(1 for s in scores if s == PROTECTED),
        steps=steps,
        view=[(t, "protected" if sc == PROTECTED else ("kept" if kp else "dropped"))
              for t, sc, kp in zip(tokens, scores, keep)],
    )
