from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import byte, rewrite

from tn.token_parser import TokenParser


ROOT = "tn.german.rules."
DATA = Path(__file__).parent / "data"


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream if "~" in line]


@pytest.fixture(scope="module", params=[True, False])
def calendar_rules(request):
    for name in ("cardinal", "ordinal", "date", "time"):
        assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name
    deterministic = request.param
    cardinal = import_module(ROOT + "cardinal").Cardinal(deterministic=deterministic)
    ordinal = import_module(ROOT + "ordinal").Ordinal(cardinal, deterministic=deterministic)
    return {
        "date": import_module(ROOT + "date").Date(cardinal, ordinal, deterministic=deterministic),
        "time": import_module(ROOT + "time").Time(cardinal, deterministic=deterministic),
    }


@pytest.mark.parametrize("name", ["date", "time"])
def test_rule_module_exists(name):
    assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name


@pytest.mark.parametrize("spoken,written", source_pairs("date"))
def test_source_date_top1(calendar_rules, spoken, written):
    assert calendar_rules["date"].normalize(written) == spoken


@pytest.mark.parametrize("spoken,written", source_pairs("time"))
def test_source_time_top1(calendar_rules, spoken, written):
    assert calendar_rules["time"].normalize(written) == spoken


@pytest.mark.parametrize("name,written,spoken,members", [
    ("date", "02.03.2003", "zweiter märz zwei tausend drei",
     {"day": "02", "month": "03", "year": "2003"}),
    ("date", "2. Mär. 2003", "zweiter märz zwei tausend drei",
     {"day": "2", "month": "Mär.", "year": "2003"}),
    ("date", "2003-03-02", "zweiter märz zwei tausend drei",
     {"year": "2003", "month": "03", "day": "02"}),
    ("time", "08:00 Uhr", "acht uhr", {"hours": "08", "minutes": "00"}),
    ("time", "00:00:00 Uhr", "null uhr null minuten null sekunden",
     {"hours": "00", "minutes": "00", "seconds": "00"}),
    ("time", "01:01:01 Uhr est", "ein uhr eine minute eine sekunde e s t",
     {"hours": "01", "minutes": "01", "seconds": "01", "zone": "est"}),
])
def test_raw_fields_and_mapping(calendar_rules, name, written, spoken, members):
    rule = calendar_rules[name]
    tagged = rule.tag(written)
    parser = TokenParser(rule.token_orders)
    parser.parse(tagged)
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert rule.verbalize(tagged) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert len(result.mappings) == 1
    mapping = result.mappings[0]
    assert mapping.input_text == written
    assert mapping.output_text == spoken
    assert (mapping.input_start, mapping.input_end) == (0, len(written))
    assert (mapping.output_start, mapping.output_end) == (0, len(spoken))


@pytest.mark.parametrize("tagged,spoken", [
    ('date { year: "2003" month: "03" day: "02" }', "zweiter märz zwei tausend drei"),
    ('date { year: "2003" month: "03" }', "märz zwei tausend drei"),
    ('date { year: "1900" }', "neunzehn hundert"),
])
def test_date_schema_order(calendar_rules, tagged, spoken):
    assert calendar_rules["date"].verbalize(tagged) == spoken


@pytest.mark.parametrize("name,attribute,written,spoken", [
    ("date", "year_graph", "1900", "neunzehn hundert"),
    ("date", "year_graph", "1980", "neunzehn achtzig"),
    ("date", "year_graph", "2003", "zwanzig drei"),
    ("date", "month_abbr_graph", "Mär.", "märz"),
    ("date", "month_graph", "03", "märz"),
    ("date", "day_graph", "02", "zweiter"),
    ("time", "number_graph", "00", "null"),
    ("time", "number_graph", "01", "eins"),
    ("time", "zone_graph", "est", "e s t"),
    ("time", "night_hour_graph", "24", "12"),
    ("time", "next_hour_graph", "23", "12"),
    ("time", "minute_to_graph", "57", "3"),
    ("time", "past_minute_graph", "15", "viertel"),
    ("time", "to_minute_graph", "45", "fünfzehn"),
])
def test_bare_semantic_graphs(calendar_rules, name, attribute, written, spoken):
    graph = getattr(calendar_rules[name], attribute)
    assert spoken in rewrite.top_rewrites(written, graph, 20)


@pytest.mark.parametrize("written,expected", [
    ("02.03.2003", {"zweiter märz zwei tausend drei", "zweiter märz zwanzig drei",
                    "zweiter dritter zwei tausend drei", "zweiter dritter zwanzig drei"}),
    ("1980", {"neunzehn achtzig", "neunzehn hundert achtzig"}),
])
def test_date_candidates(calendar_rules, written, expected):
    assert expected <= set(calendar_rules["date"].normalize(written, nbest=30))


@pytest.mark.parametrize("written,expected", [
    ("11:30 Uhr", {"elf uhr dreißig", "halb zwölf", "dreißig nach elf"}),
    ("11:45 Uhr", {"elf uhr fünf und vierzig", "viertel vor zwölf", "fünfzehn vor zwölf"}),
    ("11:57 Uhr", {"elf uhr sieben und fünfzig", "drei vor zwölf"}),
    ("12:03 Uhr", {"zwölf uhr drei", "drei nach zwölf"}),
    ("12:15 Uhr", {"zwölf uhr fünfzehn", "viertel nach zwölf", "fünfzehn nach zwölf"}),
    ("24:15 Uhr", {"vier und zwanzig uhr fünfzehn", "viertel nach zwölf", "fünfzehn nach zwölf"}),
    ("00:30 Uhr", {"null uhr dreißig", "halb eins"}),
    ("23:45 Uhr est", {"drei und zwanzig uhr fünf und vierzig e s t", "viertel vor zwölf e s t"}),
])
def test_time_candidates_and_mapping(calendar_rules, written, expected):
    rule = calendar_rules["time"]
    candidates = rule.normalize(written, nbest=30)
    assert expected <= set(candidates)
    mapped = rule.normalize_with_mapping(written, nbest=30)
    assert [result.output_text for result in mapped] == candidates
    assert all(result.mappings[0].input_text == written for result in mapped)


@pytest.mark.parametrize("name,written", [
    ("date", "00.03.2003"), ("date", "32.03.2003"), ("date", "02.13.2003"),
    ("date", "2003-00-02"), ("date", "2200-03-02"),
    ("time", "25:00 Uhr"), ("time", "08:60 Uhr"), ("time", "08:00:60 Uhr"),
    ("time", "08:7 Uhr"), ("time", "000:00 Uhr"), ("time", "2 Uhr abc"),
])
def test_bounded_calendar_fields(calendar_rules, name, written):
    assert pynini.compose(written, calendar_rules[name].tagger).start() == pynini.NO_STATE_ID


@pytest.mark.parametrize("tagged", [
    'time { hours: "25" }',
    'time { hours: "08" minutes: "60" }',
    'time { hours: "08" minutes: "00" seconds: "60" }',
])
def test_verbalizer_clock_bounds(calendar_rules, tagged):
    rule = calendar_rules["time"]
    reordered = rule.token_parser().reorder(tagged)
    assert pynini.compose(reordered, rule.verbalizer).start() == pynini.NO_STATE_ID


def test_own_token_orders(calendar_rules):
    for name, order in (("date", ["day", "month", "year"]),
                        ("time", ["hours", "minutes", "seconds", "zone"])):
        module = import_module(ROOT + name)
        assert module.TOKEN_ORDER == {name: order}
        assert calendar_rules[name].token_orders == module.TOKEN_ORDER


def test_shared_digit_is_not_mutated(calendar_rules):
    assert pynini.equivalent(byte.DIGIT, pynini.union(*"0123456789").optimize())
    assert pynini.compose("12", byte.DIGIT).start() == pynini.NO_STATE_ID
