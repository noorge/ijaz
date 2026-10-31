"""Word analysis: split each word into its parts and (optionally) tag its type.

Uses CAMeL Tools when available (better POS, real clitic analysis).
Falls back to a light rule-based splitter so the compressor always runs.
"""
import re
from functools import lru_cache

_PROCLITIC = re.compile(r"^(?P<conj>[وف])?(?P<prep>[بلك])?(?P<det>ال)?")
_ENCLITIC = re.compile(r"(?P<pron>هما|كما|هم|هن|كم|كن|نا|ها|ني|ه|ك|ي)$")

_camel = None
_camel_failed = False


def _load_camel():
    """Load CAMeL's pretrained MSA disambiguator once. Returns None if unavailable."""
    global _camel, _camel_failed
    if _camel is not None or _camel_failed:
        return _camel
    try:
        from camel_tools.disambig.mle import MLEDisambiguator
        _camel = MLEDisambiguator.pretrained()
    except Exception:
        _camel_failed = True
        _camel = None
    return _camel


def camel_available() -> bool:
    return _load_camel() is not None


@lru_cache(maxsize=50000)
def split_clitics(word: str):
    """Rule-based split: (proclitics, stem, enclitic). Keeps stems at least 3 letters."""
    m = _PROCLITIC.match(word)
    pro = m.group(0) if m else ""
    rest = word[len(pro):]
    if len(rest) < 3:  # too short: the "clitic" was probably part of the root (وزن, لبن)
        pro, rest = "", word
    e = _ENCLITIC.search(rest)
    enc = ""
    if e and len(rest) - len(e.group(0)) >= 3:
        enc = e.group(0)
        rest = rest[: -len(enc)]
    return pro, rest, enc


def pos_tags(words):
    """Return a POS tag per word ('noun', 'noun_prop', 'verb', 'adj', 'prep', ...).

    With CAMeL: real tags. Without: None (scorer uses heuristics instead).
    """
    dis = _load_camel()
    if dis is None:
        return None
    try:
        out = dis.disambiguate(list(words))
        tags = []
        for d in out:
            if d.analyses:
                tags.append(d.analyses[0].analysis.get("pos", "unk"))
            else:
                tags.append("unk")
        return tags
    except Exception:
        return None
