"""Baselines we compare against, all matched to the same token budget."""
import random
import re

from .tokens import count_tokens, word_cost


def random_drop(text, keep_rate, seed=0):
    """Delete random words until the budget is met (the 'is it better than luck?' check)."""
    words = text.split()
    budget = keep_rate * count_tokens(text)
    order = list(range(len(words)))
    random.Random(seed).shuffle(order)
    keep = [True] * len(words)
    current = count_tokens(text)
    for i in order:
        if current <= budget:
            break
        keep[i] = False
        current -= word_cost(words[i])
    return " ".join(w for w, k in zip(words, keep) if k)


def truncate(text, keep_rate):
    """Keep the beginning of the passage only."""
    words, out, budget = text.split(), [], keep_rate * count_tokens(text)
    for w in words:
        if count_tokens(" ".join(out + [w])) > budget:
            break
        out.append(w)
    return " ".join(out)


_LL2 = None


def llmlingua2(text, keep_rate):
    """Microsoft LLMLingua-2 (English-supervised). Needs: pip install llmlingua"""
    global _LL2
    if _LL2 is None:
        import torch
        from llmlingua import PromptCompressor
        _LL2 = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            use_llmlingua2=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
    out = _LL2.compress_prompt(text, rate=keep_rate, force_tokens=["\n", "?", "؟"])
    return re.sub(r" {2,}", " ", out["compressed_prompt"]).strip()
