from importlib import import_module, util
from pathlib import Path

import pytest
from pynini.lib import rewrite

from tn.german.rules.cardinal import Cardinal
from tn.german.rules.time import Time as TNTime


DATA = Path(__file__).parent / "data" / "test_cases_time.txt"
with DATA.open(encoding="utf-8", newline="") as stream:
    CASES = [tuple((line[:-1] if line.endswith("\n") else line).split("~")) for line in stream]


def test_native_time_exists():
    assert util.find_spec("itn.german.rules.time") is not None


@pytest.fixture(scope="module")
def rule():
    return import_module("itn.german.rules.time").Time(TNTime(Cardinal(False), False))


@pytest.mark.parametrize("spoken,written", CASES)
def test_source_time_components(rule, spoken, written):
    for suffix in (" mittags", " nachts"):
        if spoken.endswith(suffix):
            spoken, written = spoken[:-len(suffix)], written[:-len(suffix)]
    assert rule.normalize(spoken) == written
    result = rule.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    assert result.mappings[0].input_text == spoken
    assert result.mappings[0].output_text == written


@pytest.mark.parametrize("spoken,written,members", [
    ("vierundzwanziguhrzweiundzwanzigest", "24:22 Uhr est",
     {"kind": "clock", "hours": "vierundzwanzig", "minutes": "zweiundzwanzig", "zone": "est"}),
    ("viertel vor zwölf", "11:45 Uhr", {"kind": "to", "hours": "zwölf", "minutes": "viertel"}),
    ("halb zwölf", "11:30 Uhr", {"kind": "half", "hours": "zwölf"}),
    ("ein uhr eine minute eine sekunde e s t", "01:01:01 Uhr est",
     {"kind": "clock", "hours": "ein", "minutes": "eine minute", "seconds": "eine sekunde", "zone": "e s t"}),
    ("neunundneunziguhr", "99 Uhr", {"kind": "hour", "hours": "neunundneunzig"}),
    ("zwei uhr neunundneunzig", "02:99 Uhr",
     {"kind": "clock", "hours": "zwei", "minutes": "neunundneunzig"}),
])
def test_raw_fields(rule, spoken, written, members):
    parser = rule.token_parser()
    parser.parse(rule.tag(spoken))
    assert parser.tokens[0].members == members
    assert rule.normalize(spoken) == written


def test_colloquial_night_alternatives(rule):
    assert set(rule.normalize("halb zwölf", nbest=10)) == {"11:30 Uhr", "23:30 Uhr"}
    assert set(rule.normalize("viertel nach zwölf", nbest=10)) == {"00:15 Uhr", "12:15 Uhr", "24:15 Uhr"}


def test_reordered_raw_tokens(rule):
    parser = rule.token_parser()
    parser.parse(rule.tag("drei vor zwölf"))
    assert parser.tokens[0].members == {"kind": "to", "hours": "zwölf", "minutes": "drei"}
    assert rule.verbalize(parser.reorder(rule.tag("drei vor zwölf"))) == "11:57 Uhr"


@pytest.mark.parametrize("spoken", ["EIN uhr", "eins uhr", "hundert uhr", "ein uhr eins minute"])
def test_source_exclusions(rule, spoken):
    with pytest.raises(rewrite.Error):
        rewrite.one_top_rewrite(spoken, rule.tagger)
