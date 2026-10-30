from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import rewrite

from tn.token_parser import TokenParser


DATA = Path(__file__).parent / "data" / "test_cases_date.txt"


def source_pairs():
    with DATA.open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream if "~" in line]


def test_native_date_class_exists():
    name = "itn.german.rules.date"
    assert util.find_spec(name) is not None, "missing native German ITN Date"
    assert hasattr(import_module(name), "Date")


@pytest.fixture(scope="module")
def date():
    from itn.german.rules.cardinal import Cardinal
    from itn.german.rules.date import Date
    from tn.german.rules.cardinal import Cardinal as TNCardinal
    from tn.german.rules.date import Date as TNDate
    from tn.german.rules.ordinal import Ordinal as TNOrdinal

    tn_cardinal = TNCardinal(deterministic=False)
    tn_ordinal = TNOrdinal(tn_cardinal, deterministic=False)
    tn_date = TNDate(tn_cardinal, tn_ordinal, deterministic=False)
    return Date(Cardinal(tn_cardinal), tn_date, tn_ordinal)


@pytest.fixture(scope="module")
def inverse_normalizer():
    from itn.german.inverse_normalizer import InverseNormalizer

    return InverseNormalizer(cache_dir=False)


@pytest.mark.parametrize("spoken,written", source_pairs())
def test_source_date(inverse_normalizer, spoken, written):
    assert inverse_normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken,written,members", [
    ("vier und zwanzigster juli zwei tausend dreizehn", "24. Jul. 2013",
     {"day": "vier und zwanzigster", "month": "juli", "year": "zwei tausend dreizehn"}),
    ("erster januar", "1. Jan.", {"day": "erster", "month": "januar"}),
    ("zweiter dritter", "02.03.", {"day": "zweiter", "month": "dritter"}),
    ("zweiter dritter neunzehnhundertachtzig", "02.03.1980",
     {"day": "zweiter", "month": "dritter", "year": "neunzehnhundertachtzig"}),
    ("januar zweitausendneun", "Jan. 2009", {"month": "januar", "year": "zweitausendneun"}),
    ("neunzehnachtzig", "1980", {"year": "neunzehnachtzig"}),
    ("zwanzig zwanzig", "2020", {"year": "zwanzig zwanzig"}),
])
def test_raw_fields_mapping_and_nbest(date, spoken, written, members):
    tagged = date.tag(spoken)
    parser = TokenParser(date.token_orders)
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "date"
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert date.normalize(spoken) == written
    assert date.verbalize(tagged) == written
    candidates = date.normalize(spoken, nbest=5)
    assert candidates[0] == written
    results = date.normalize_with_mapping(spoken, nbest=5)
    assert [result.output_text for result in results] == candidates
    for result in results:
        assert result.input_text == spoken
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.kind == "replace"
        assert mapping.token_type == "date"
        assert mapping.input_text == spoken
        assert mapping.output_text == result.output_text
        assert (mapping.input_start, mapping.input_end) == (0, len(spoken))
        assert (mapping.output_start, mapping.output_end) == (0, len(result.output_text))


@pytest.mark.parametrize("spoken,written", [
    ("erster februar", "1. Feb."),
    ("einunddreißigster dezember", "31. Dez."),
    ("neunter zwölfter zwei tausend neun", "09.12.2009"),
    ("elfhundert", "1100"),
    ("zehn eins", "1001"),
    ("zwanzigeins", "2001"),
    ("zwei tausend einhundertneunundneunzig", "2199"),
    ("nullter nullter", "00.00."),
    ("einundvierzigster dreizehnter", "41.13."),
    ("einhundertster januar", "100. Jan."),
    ("neunundneunzigster neunundneunzigster", "99.99."),
])
def test_source_semantic_language(date, spoken, written):
    assert date.normalize(spoken) == written


@pytest.mark.parametrize("attribute,spoken,written", [
    ("day_graph", "erster", "1"),
    ("day_graph", "vier und zwanzigster", "24"),
    ("numeric_month_graph", "dritter", "3"),
    ("named_month_graph", "juli", "Jul."),
    ("month_graph", "januar", "Jan."),
    ("year_graph", "neunzehnhundertachtzig", "1980"),
    ("year_graph", "zwanzigzwanzig", "2020"),
])
def test_bare_semantic_graphs(date, attribute, spoken, written):
    assert written in rewrite.top_rewrites(spoken, getattr(date, attribute), 10)


@pytest.mark.parametrize("spoken,written", [
    ("zweiter dritter", "02.03."),
    ("vierzehnter januar", "14. Jan."),
    ("juli zwei tausend dreizehn", "Jul. 2013"),
    ("neunzehn achtzig", "1980"),
])
def test_flat_raw_fields(date, spoken, written):
    fields = rewrite.one_top_rewrite(spoken, date.fields_tagger)
    assert "{" not in fields and "preserve_order" not in fields
    assert rewrite.one_top_rewrite(fields, date.fields_verbalizer) == written


@pytest.mark.parametrize("spoken", [
    "januar", "januarzweitausendneun", "neunzehn achtziger", "ein tausend acht",
    "neunzehn", "vierundzwanzigsten juli", "vierundzwanzigste juli", "zweiter März",
    "zweiter mär", "zweiter d r i t t e r", "zweiter", "dritter neunzehnachtzig",
    "zwei tausend zweihundert", "neunhundertneunundneunzig", "acht hundert",
    "einhundertster dritter", "zweiter einhundertster",
])
def test_invalid_and_pipeline_context_inputs(date, spoken):
    assert pynini.compose(pynini.accep(pynini.escape(spoken)), date.tagger).start() == pynini.NO_STATE_ID


def test_order_contract(date):
    from itn.german.rules.date import TOKEN_ORDER

    assert TOKEN_ORDER == {"date": ["day", "month", "year"]}
    assert date.token_orders == TOKEN_ORDER
    assert date.ordertype == "itn"
    assert date.verbalize('date { year: "neunzehnachtzig" month: "juli" day: "zweiter" }') == "2. Jul. 1980"
