from functools import lru_cache
from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import byte, rewrite

from tn.token_parser import TokenParser


ROOT = "tn.german.rules."
DATA = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def financial_rules(deterministic=True):
    modules = {}
    for name in ("cardinal", "decimal", "ordinal", "fraction", "money", "measure"):
        assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name
        modules[name] = import_module(ROOT + name)
    cardinal = modules["cardinal"].Cardinal(deterministic=deterministic)
    decimal = modules["decimal"].Decimal(cardinal, deterministic=deterministic)
    ordinal = modules["ordinal"].Ordinal(cardinal, deterministic=deterministic)
    fraction = modules["fraction"].Fraction(cardinal, ordinal, deterministic=deterministic)
    return {
        "money": modules["money"].Money(cardinal, decimal, deterministic=deterministic),
        "measure": modules["measure"].Measure(cardinal, decimal, fraction, deterministic=deterministic),
    }


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\n").split("~")) for line in stream if "~" in line]


@pytest.mark.parametrize("name", ["money", "measure"])
def test_rule_modules_exist(name):
    assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("spoken,written", source_pairs("money"))
def test_source_money(deterministic, spoken, written):
    assert financial_rules(deterministic)["money"].normalize(written) == spoken


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("spoken,written", source_pairs("measure"))
def test_source_measure(deterministic, spoken, written):
    assert financial_rules(deterministic)["measure"].normalize(written) == spoken


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("name,written,spoken,members", [
    ("money", "€1,000", "ein euro", {"currency": "€", "kind": "major", "integer_part": "1", "fractional_part": "000"}),
    ("money", "$2,2010", "zwei komma zwei null eins dollar",
     {"currency": "$", "kind": "decimal", "integer_part": "2", "fractional_part": "2010"}),
    ("money", "£0,0100", "ein penny",
     {"currency": "£", "kind": "minor", "integer_part": "0", "fractional_part": "0100"}),
    ("money", "€-1,20 millionen", "minus eins komma zwei null millionen euro",
     {"currency": "€", "kind": "decimal", "negative": "-", "integer_part": "1",
      "fractional_part": "20", "quantity": "millionen"}),
    ("measure", "-1 h", "minus ein stunde", {"kind": "unit", "negative": "-", "integer": "1", "units": "h"}),
    ("measure", "a-8", "a acht", {"kind": "alpha", "prefix": "a", "integer": "8"}),
    ("measure", "8-a", "acht a", {"kind": "alpha", "integer": "8", "units": "a"}),
    ("measure", "a-1,20", "a eins komma zwei null",
     {"kind": "alpha", "prefix": "a", "integer_part": "1", "fractional_part": "20"}),
    ("measure", "1,2-a", "eins komma zwei a",
     {"kind": "alpha", "integer_part": "1", "fractional_part": "2", "units": "a"}),
    ("measure", "1 1/2 ft", "ein ein halb fuß",
     {"kind": "unit", "integer_part": "1", "numerator": "1", "denominator": "2", "units": "ft"}),
    ("measure", "2km/h", "zwei kilometer pro stunde", {"kind": "unit", "integer": "2", "units": "km/h"}),
    ("measure", "2X", "zwei X", {"kind": "times", "integer": "2", "units": "X"}),
    ("measure", "2,00x", "zwei komma null null x",
     {"kind": "times", "integer_part": "2", "fractional_part": "00", "units": "x"}),
])
def test_raw_fields_and_mapping(deterministic, name, written, spoken, members):
    rule = financial_rules(deterministic)[name]
    tagged = rule.tag(written)
    parser = TokenParser(rule.token_orders)
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert "currency_min" not in tagged and "preserve_order" not in tagged
    assert rule.verbalize(tagged) == spoken
    assert rule.normalize(written) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert len(result.mappings) == 1
    mapping = result.mappings[0]
    assert mapping.input_text == written and mapping.output_text == spoken
    assert (mapping.input_start, mapping.input_end) == (0, len(written))
    assert (mapping.output_start, mapping.output_end) == (0, len(spoken))


def test_money_nbest_minor_alternatives_and_mapping():
    rule = financial_rules(False)["money"]
    expected = {
        "zwei dollar zwanzig cent", "zwei dollar und zwanzig cent", "zwei dollar zwanzig",
        "zwei us dollar zwanzig cent", "zwei us dollar und zwanzig cent", "zwei us dollar zwanzig",
    }
    candidates = rule.normalize("$2,20", nbest=20)
    assert candidates[0] == "zwei dollar zwanzig cent"
    assert expected == set(candidates)
    mapped = rule.normalize_with_mapping("$2,20", nbest=20)
    assert [item.output_text for item in mapped] == candidates
    assert all(item.mappings[0].input_text == "$2,20" for item in mapped)


@pytest.mark.parametrize("name", ["money", "measure"])
def test_long_amount_schema_reordering(name):
    rule = financial_rules()[name]
    number = "999999999999999999999999"
    written = "€" + number if name == "money" else "a-" + number
    members = {"currency": "€", "kind": "major", "integer_part": number} if name == "money" else {
        "kind": "alpha", "prefix": "a", "integer": number,
    }
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert parser.tokens[0].members == members
    reordered = parser.reorder(rule.tag(written))
    if name == "money":
        assert reordered.index("integer_part") < reordered.index("currency")
    else:
        assert reordered.index("prefix") < reordered.index("integer")
    result = rule.normalize_with_mapping(written)
    assert result.output_text == rule.normalize(written)
    assert result.mappings[0].input_text == written
    shuffled = name + " { " + " ".join('{}: "{}"'.format(key, value) for key, value in reversed(list(members.items()))) + " }"
    assert rule.verbalize(shuffled) == result.output_text


@pytest.mark.parametrize("attribute,written,spoken", [
    ("unit_singular_graph", "h", "stunde"),
    ("unit_plural_graph", "h", "stunden"),
    ("unit_plural_graph", "kwh", "kilowattstunden"),
    ("unit_denominator_graph", "/km²", "pro quadrat kilometer"),
    ("units_singular_graph", "km/h", "kilometer pro stunde"),
    ("units_plural_graph", "oz/h", "unzen pro stunde"),
    ("units_graph", "oz", "unze"),
    ("units_graph", "oz", "unzen"),
])
def test_bare_unit_semantic_api(attribute, written, spoken):
    rule = financial_rules()["measure"]
    assert spoken in rewrite.top_rewrites(written, getattr(rule, attribute), 20)


@pytest.mark.parametrize("name,written", [
    ("money", "€-2"), ("money", "-€2"), ("money", "€2,"), ("money", "€0,00x"),
    ("measure", "a--8"), ("measure", "-8-a"), ("measure", "8/"), ("measure", "2  kg"),
])
def test_source_unsupported_inputs(name, written):
    rule = financial_rules()[name]
    assert pynini.compose(pynini.accep(pynini.escape(written)), rule.tagger).start() == pynini.NO_STATE_ID


def test_schema_and_shared_byte_graphs():
    before = byte.DIGIT.copy()
    for deterministic in (True, False):
        for name, rule in financial_rules(deterministic).items():
            assert rule.token_orders == import_module(ROOT + name).TOKEN_ORDER
            assert rule.token_orders[name][0] == "kind"
            assert rule.normalize("") == ""
            assert rule.normalize("", nbest=2) == [""]
    assert pynini.equivalent(before, byte.DIGIT)
    assert pynini.compose("12", byte.DIGIT).start() == pynini.NO_STATE_ID
