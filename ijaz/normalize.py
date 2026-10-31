"""Layer 1: cleaning. Near-lossless savings before any word is dropped."""
import re

# Harakat, tanween, shadda, sukun, superscript alef, Quranic marks
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_MULTI_SPACE = re.compile(r"[ \t ]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([،؛؟.!:,;?)\]])")
_REPEATED_PUNCT = re.compile(r"([!؟?.،])\1+")


def normalize(text: str, keep_diacritics: bool = False, unify_letters: bool = False) -> str:
    """Strip diacritics and tatweel, tidy whitespace and punctuation.

    keep_diacritics: set True for Quranic text or poetry, where harakat carry meaning.
    unify_letters: also map أ/إ/آ->ا and ى->ي. Off by default: it saves little and
    can change meaning in rare cases.
    """
    if not keep_diacritics:
        text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    if unify_letters:
        text = re.sub("[أإآ]", "ا", text).replace("ى", "ي")
    text = _REPEATED_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_scoring(text: str) -> str:
    """Aggressive normalization used ONLY to compare answers, never sent to a model."""
    text = _DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = re.sub("[أإآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"[^\w\s]", " ", text)
    return _MULTI_SPACE.sub(" ", text).strip()
