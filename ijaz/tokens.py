"""Token counting. Budgets use GPT's o200k tokenizer as a shared ruler.
Real billed tokens are taken from the API responses in the evaluation."""
from functools import lru_cache

try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
except Exception:  # offline fallback: rough Arabic average
    _enc = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is None:  # rough estimate: ~1 token per 3.5 Arabic characters
        return sum(1 + len(w) // 4 for w in text.split())
    return len(_enc.encode(text))


@lru_cache(maxsize=100000)
def word_cost(word: str) -> int:
    """Approximate tokens a word adds inside running text."""
    return max(1, count_tokens(" " + word))
