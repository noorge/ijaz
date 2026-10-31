# إيجاز · Ijaz

Arabic context compression for any LLM. Ijaz shortens Arabic text before it is sent to a model, so the model reads fewer tokens and still finds the facts. It needs no question, so you compress once and reuse the result (documents, system prompts, chat history).

Built for the HUMAIN hackathon, track "Deliver AI with Less Compute".

## Why

- **Arabic token tax.** The same passages cost more tokens in Arabic than in English: 2.78× on Claude, 1.62× on Qwen2.5, 1.37× on GPT (o200k). ALLaM fixes this at the tokenizer (0.97×), but most Arabic apps call other models. (`scripts/token_tax.py`, `results/token_tax.csv`)
- **English-trained compressors break Arabic.** "Lost in Compression" (2026) found that at a third of the tokens, English keeps 57–62% of its usefulness and Arabic only 16–29%.

## How it works

Ijaz is built on the classical Arabic rules of hadhf (الحذف, ellipsis): drop only what the reader can recover from a clue (دليل) in the text.

1. **Teacher.** GPT-5.4-mini labels 5,000 Arabic Wikipedia passages with a prompt built on the classical rules of hadhf (`data/teacher_prompt.txt`, `scripts/teacher_label.py`). Each word gets a level 0–3: how safely it can be dropped.
2. **Student.** CAMeLBERT-MSA (110M) learns those labels as a word tagger (`scripts/train_student.py`).
3. **Answerness.** A second CAMeLBERT learns which words tend to hold answers, trained on TyDiQA-GoldP Arabic with ARCD articles excluded (`scripts/train_answerness.py`).
4. **Compress.** Scores are combined, and the lowest-scoring words are deleted until the text fits the token budget. Digits, negations, exception, condition and contrast words (and the word after each) are never deleted (`ijaz/learned.py`, `compress_final`). About 50 ms per passage on a GPU.

```python
from ijaz.learned import compress_final, plan_final
short = compress_final(text, keep_rate=0.33)          # compressed text
p = plan_final(text, keep_rate=0.33)                   # full details per word
print(p["original_tokens"], "->", p["final_tokens"])
```

## Results

200 random native Arabic questions from ARCD, answered by GPT-5.4-mini and Claude Haiku 4.5. Correct = F1 ≥ 0.5 or the answer contains the gold answer.

| Keep | GPT · LLMLingua-2 | GPT · Ijaz | Claude · LLMLingua-2 | Claude · Ijaz | Verdict (paired sign test) |
|---|---|---|---|---|---|
| 75% | 78.5% | 77.0% | 82.5% | 78.5% | LLMLingua-2 as good or better (30 vs 41, p = 0.24) |
| 50% | 62.5% | 66.0% | 61.0% | 63.5% | Tie (63 vs 51, p = 0.30) |
| **33%** | 47.5% | **56.5%** | 44.0% | **49.5%** | **Ijaz wins (71 vs 42, p = 0.008)** |

Full text: 82.5% / 89.5%. No text: 28.5% / 25.0%. Ijaz uses 2 × 110M parameters against LLMLingua-2's 560M. Raw answers and the summary are in `results/`.

Limits: at light compression LLMLingua-2 is as good or better; formal Arabic (MSA) only. See `FUTURE_WORK.md`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
camel_data -i light
copy .env.example .env          # add your API keys
```

Trained models are not in the repo (too large). Rebuild them with `scripts/train_student.py --labels data/teacher_labels.jsonl --out models/ijaz-student-5k` and `scripts/train_answerness.py`.

- Evaluate: `python scripts/run_eval.py --n 200 --methods full none llmlingua2 ijaz_final`, then `python scripts/summarize.py`
- Demo: `streamlit run app.py`

## Related work

LLMLingua-2 (baseline), LSPM (query-aware Arabic RAG pruning), XProvence, tokenizer-level fixes (ALLaM, AraToken). Ijaz differs by being question-free, word-level, Arabic-specific and tested on native Arabic text.
