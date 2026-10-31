"""Run: python -m pytest tests -q"""
import json

from ijaz import compress, normalize
from ijaz.lexicon import NEGATIONS


def test_diacritics_removed():
    assert normalize("مُحَمَّدٌ") == "محمد"


def test_negation_and_its_verb_survive():
    t = "ومن الجدير بالذكر أن الشركة لم تحقق أرباحاً في عام 2019، ولكنها حققت أرباحاً كبيرة."
    out = compress(t, 0.5).text
    assert "لم تحقق" in out and "2019" in out


def test_numbers_survive():
    out = compress("وصل عدد الزوار إلى 15000 زائر خلال ثلاثة أيام فقط من الافتتاح الرسمي للمعرض.", 0.4).text
    assert "15000" in out and "ثلاثة" in out


def test_budget_respected_on_suite():
    for p in json.load(open("data/negation_probe.json", encoding="utf-8")):
        for rate in (0.75, 0.5):
            r = compress(p["context"], rate)
            assert r.achieved_rate <= rate + 0.02, (p["id"], rate, r.achieved_rate)
            negs = [w for w in p["context"].split() if w in NEGATIONS]
            assert all(n in r.text for n in negs), p["id"]
