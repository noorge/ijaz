# Ijaz demo: voiceover script

About 4 minutes. Lines in **bold** are what you say. Lines in *[brackets]* are what you do on screen.
`⏸` = take a breath, let the screen catch up. Don't rush the pauses: silence while something animates is fine.

Before recording:
- Run `streamlit run app.py`, wait for it to load once, then refresh so everything is warm.
- In the Live tab, keep the pre-filled Mecca passage and set Keep to **33%**.
- Close other tabs and notifications. Zoom the browser to ~110% so the Arabic is easy to read.

---

## 1. Opening (header cards)

*[Start on the top of the page, the four cards visible.]*

**This is Ijaz. إيجاز.**

⏸

**Arabic is expensive to send to AI models. The same content costs 1.37 times more tokens than English on GPT's tokenizer, and 2.78 times more on Claude's.**

⏸

**HUMAIN's ALLaM already fixed this at the tokenizer: 0.97. But most Arabic apps still run on other models. Ijaz is for them.**

⏸

**It sits between the app and the model, and shortens the Arabic text before it's sent.**

⏸ ⏸

---

## 2. Live demo: the process

*[Click "⚡ Live demo". The Mecca passage and question are already filled in. Keep = 33%.]*

**Here's a short Arabic passage about an official delegation's tour. The question: did the delegation visit Mecca?**

⏸

**I'll ask Ijaz to keep only a third of the tokens.**

⏸

*[Click "① Read".]*

**AI models read text in small pieces called tokens, and you pay for every one.**

⏸

**This passage is seventy-three tokens.**

⏸

**Keeping a third means we can only send twenty-four. That's our budget.**

⏸ ⏸

*[Click "② Teacher · الحذف".]*

**So which words go? Arabic already has an answer for that.**

⏸

**Classical Arabic rhetoric has rules for leaving words out. It's called الحذف.**

⏸

**The main rule is simple: you may drop a word only if the reader can still recover it.**

⏸

**A phrase like «من الجدير بالذكر أن», "it's worth mentioning that", can go. The reader loses nothing.**

⏸

**But a name or a date can't be recovered. Drop it, and it's gone.**

⏸ ⏸

**We wrote these rules into instructions for GPT, and had it mark five thousand Arabic passages: which words can go, and how safely.**

⏸

**GPT is too expensive to run on every request, so we trained a small Arabic model to copy its judgement.**

⏸

**That's what you see here. Blue means keep. Red is safe to drop. Orange and purple go later.**

⏸ ⏸

*[Click "③ Answerness".]*

**But rules only tell us what's safe to drop. Not what's worth keeping.**

⏸

**So we added a second small model. We call it answerness.**

⏸

**It learned from ten thousand Arabic passages where real people asked questions and marked the answers. Now it knows which words carry the facts people ask about. The brighter the word, the more it matters.**

⏸

**Ijaz never sees the question. It just knows what usually matters.**

⏸ ⏸

*[Click "④ Combine & protect".]*

**Both signals become one score per word. And some words are never deleted: numbers, and negation, like «لم». Drop those and the meaning flips.**

⏸ ⏸

*[Click "⑤ Delete to budget", then "▶ Play the deletion". Let it run. Say nothing until it stops.]*

⏸ ⏸ ⏸

**Lowest score first, until it fits.**

⏸

---

## 3. Side by side

*[Scroll to "Ijaz vs LLMLingua-2 at the same budget".]*

**Now the same passage through LLMLingua-2, Microsoft's compressor, at the same budget.**

⏸

**Ijaz kept «إلا أنه لم يزر مكة المكرمة»: "but it did not visit Mecca."**

⏸

**LLMLingua-2 dropped the negation, and kept "visited Riyadh… Mecca."**

⏸ ⏸

**Same budget. Ijaz kept the answer. LLMLingua-2 lost it.**

⏸

**That's one example. Now let's see what the models actually answered on real test questions.**

⏸ ⏸

---

## 4. A real test question

*[Click "📚 Test examples". Keep = 33%. Filter: "Ijaz right, LLMLingua-2 wrong". Pick «ماهو الإسم الكامل للعلامة ابن خلدون؟».]*

**These are real questions from our test set, native Arabic, with the answers the models actually gave during evaluation.**

⏸

**Ibn Khaldun's full name.**

⏸

*[Scroll to the answers.]*

**With Ijaz, both GPT and Claude get it right. With LLMLingua-2, both get it wrong.**

⏸

**Fittingly, Ibn Khaldun himself warned that abridging too much harms understanding.**

⏸ ⏸

*[Switch the filter to "LLMLingua-2 right, Ijaz wrong" for a second.]*

**And to be fair, there are questions where it's the other way round. We show those too.**

⏸ ⏸

---

## 5. Results

*[Click "📊 Results".]*

**Two hundred random Arabic questions, two models.**

⏸

**At a third of the tokens, Ijaz keeps more correct answers: 56.5 percent against 47.5 on GPT, 49.5 against 44 on Claude.**

⏸

**Question by question, 71 that only Ijaz got right, 42 that only LLMLingua-2 did. That's not luck: p equals 0.008.**

⏸

**At half the tokens, they're tied. At three quarters, LLMLingua-2 is as good or slightly better. Ijaz is built for heavy compression.**

⏸

**And Ijaz does it with two small models, two hundred and twenty million parameters, against five hundred and sixty million. On an ordinary gaming GPU, about fifty milliseconds per passage.**

⏸ ⏸

---

## 6. Close

*[Scroll back to the top cards.]*

**Arabic readers have always filled in what was left unsaid. Ijaz uses the rules of إيجاز to do the same for machines.**

⏸

**Less compute, same answers, in Arabic first.**

⏸ ⏸

*[Stop recording.]*

---

### If something goes wrong while recording
- **A step is slow to appear:** wait a second in silence. The first passage after starting the app is the slowest.
- **You stumble on a line:** pause, breathe, and say it again. Cut the mistake in editing.
- **Numbers to remember:** 1.37× / 2.78× / ALLaM 0.97× · 56.5 vs 47.5 · 49.5 vs 44 · 71 vs 42, p = 0.008 · 220M vs 560M · ~50 ms.
