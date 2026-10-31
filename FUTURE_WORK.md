# Ijaz: future work

Ideas we decided to keep for later, with what we already know about each.

## 1. ALLaM as the teacher
- **Tried:** ALLaM-7B-Instruct (4-bit, local) as the labelling teacher on 5 passages.
- **Result:** it did not read the passages. It copied the word numbers from the worked example in the prompt
  (0-3, 5, 9-10, 11, 15, 20-25, 26-27) and then marked every second word as حشو.
  It also tried to delete facts like «4.6 مليار سنة» and «1 يناير 2002».
- **Next:** a larger ALLaM, or fine-tune ALLaM on our 5,000 GPT labels, so the whole pipeline runs on HUMAIN's own model
  with no external API.

## 2. Arabic professors label the data
- Linguists label a few hundred passages with the same الحذف framework (levels + reasons).
- Two professors on the same passages give the real ceiling (how often experts agree with each other).
- Grade the GPT teacher against the experts; if it is far off, train on the expert labels instead.
- Result: a linguist-annotated Arabic compression dataset, which does not exist today and is directly useful for ALLaM.

## 3. A cleaner teacher
- **Self-consistency check (done, `scripts/teacher_consistency.py`):** the teacher relabelled 100 passages
  (11,164 words) it had already labelled. Teacher vs itself: 74.1% keep-vs-delete, 60.1% exact level.
  Student vs teacher: 70.4% / 59.2%. The student is about 4 points below the teacher's own consistency,
  so more labels from this teacher add little (1k → 5k barely moved). Cleaner labels are the way up.
- **Consensus labels:** label each passage twice and keep only the words where both runs agree.
- **Stronger teacher** on fewer passages.

## 4. Loosen the teacher prompt for machine readers
- Finding: the teacher keeps grammatical glue (في، من، هو) and deletes names and فضلة. LLMLingua-2 does the opposite,
  and an LLM reader restores glue on its own: for a machine reader, glue always has a دليل عقلي.
- Revise the prompt: glue goes to level 1; names, numbers and specific nouns are deleted last. Relabel and retrain.

## 5. The student explains itself
- Today the student learns only the level (0-3); the teacher's reason (حشو، تكرار، فضلة…) is saved but not learned.
- Train a 13-way head (one slot per reason) so every deleted word in the demo can show *why* it was deleted.

## 6. Summarization as a comparison
- Have GPT summarize each passage to the same token budget and run the same questions.
- Expected: reads better, but costs a full LLM generation per passage (the opposite of the goal) and can change facts.
  Worth showing as a comparison, not as a competitor.

## 7. Dialects
- Ijaz is MSA only. Gulf / Egyptian / Levantine text needs dialect-aware teacher prompts and training passages.
