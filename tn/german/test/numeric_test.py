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
def numeric_rules(deterministic):
    modules = {}
    for name in ("cardinal", "decimal", "ordinal", "fraction"):
        assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name
        modules[name] = import_module(ROOT + name)
    cardinal = modules["cardinal"].Cardinal(deterministic=deterministic)
    ordinal = modules["ordinal"].Ordinal(cardinal, deterministic=deterministic)
    return {
        "cardinal": cardinal,
        "ordinal": ordinal,
        "decimal": modules["decimal"].Decimal(cardinal, deterministic=deterministic),
        "fraction": modules["fraction"].Fraction(cardinal, ordinal, deterministic=deterministic),
    }


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream if "~" in line]


@pytest.mark.parametrize("name", ["cardinal", "decimal", "ordinal", "fraction"])
def test_rule_modules_exist(name):
    assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("name,written,spoken,members", [
    ("cardinal", "-123", "minus ein hundert drei und zwanzig", {"negative": "-", "integer": "123"}),
    ("cardinal", "1000000", "eine million", {"integer": "1000000"}),
    ("decimal", "-60,2400", "minus sechzig komma zwei vier null null",
     {"negative": "-", "integer_part": "60", "fractional_part": "2400"}),
    ("decimal", "0,2 millionen", "null komma zwei millionen",
     {"integer_part": "0", "fractional_part": "2", "quantity": "millionen"}),
    ("decimal", "460 millionen", "vier hundert sechzig millionen",
     {"integer_part": "460", "quantity": "millionen"}),
    ("fraction", "1 1/2", "ein ein halb", {"integer_part": "1", "numerator": "1", "denominator": "2"}),
    ("fraction", "-1 / 22", "minus ein zwei und zwanzigstel",
     {"negative": "-", "numerator": "1", "denominator": "22"}),
])
def test_raw_fields_and_mapping(deterministic, name, written, spoken, members):
    rule = numeric_rules(deterministic)[name]
    tagged = rule.tag(written)
    parser = TokenParser(rule.token_orders)
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert rule.normalize(written) == spoken
    assert rule.verbalize(tagged) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert len(result.mappings) == 1
    mapping = result.mappings[0]
    assert mapping.input_text == written
    assert mapping.output_text == spoken
    assert (mapping.input_start, mapping.input_end) == (0, len(written))
    assert (mapping.output_start, mapping.output_end) == (0, len(spoken))


@pytest.mark.parametrize("spoken,written", source_pairs("cardinal"))
def test_source_cardinal(spoken, written):
    assert numeric_rules(True)["cardinal"].normalize(written) == spoken


@pytest.mark.parametrize("spoken,written", source_pairs("fraction"))
def test_source_fraction(spoken, written):
    assert numeric_rules(True)["fraction"].normalize(written) == spoken


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("written,base", [("1.", "ers"), ("21.", "ein und zwanzigs"), ("121tem", "ein hundert ein und zwanzigs")])
def test_ordinal_all_endings_and_raw_integer(deterministic, written, base):
    rule = numeric_rules(deterministic)["ordinal"]
    expected = {base + ending for ending in ("ten", "tem", "ter", "tes", "te")}
    candidates = rule.normalize(written, nbest=30)
    assert expected <= set(candidates)
    mapped = rule.normalize_with_mapping(written, nbest=30)
    assert [result.output_text for result in mapped] == candidates
    parser = TokenParser(rule.token_orders)
    parser.parse(rule.tag(written))
    integer = written[:-3] if written.endswith("tem") else written[:-1]
    suffix = "tem" if written.endswith("tem") else "."
    assert parser.tokens[0].members == {"integer": integer, "suffix": suffix}
    assert rule.verbalize(rule.tag(written)) in candidates
    assert all(result.mappings[0].input_text == written for result in mapped)


@pytest.mark.parametrize("name,attribute,written,spoken", [
    ("cardinal", "graph", "123", "ein hundert drei und zwanzig"),
    ("cardinal", "digits", "1", "eins"),
    ("cardinal", "digits", "0", "null"),
    ("cardinal", "hundreds", "999", "neun hundert neun und neunzig"),
    ("cardinal", "two_digits", "21", "ein und zwanzig"),
    ("cardinal", "sign", "-", "minus "),
    ("ordinal", "stem", "eins", "ers"),
    ("ordinal", "suffix", "ein hundert eins", "ein hundert erster"),
    ("ordinal", "graph", "101", "ein hundert erster"),
    ("decimal", "integer_graph", "12", "zwölf"),
    ("decimal", "fractional_graph", "010", "null eins null"),
    ("decimal", "quantity_graph", "millionen", "millionen"),
    ("fraction", "integer_graph", "1", "ein"),
    ("fraction", "numerator_graph", "1", "ein"),
    ("fraction", "denominator_graph", "2", "halbe"),
    ("fraction", "denominator_graph", "22", "zwei und zwanzigstel"),
])
def test_public_semantic_graphs(name, attribute, written, spoken):
    rule = numeric_rules(True)[name]
    assert spoken in rewrite.top_rewrites(written, getattr(rule, attribute), 20)


@pytest.mark.parametrize("name,written,spoken", [
    ("decimal", "-1,05 millionen", "minus eins komma null fünf millionen"),
    ("fraction", "-3 1/2", "minus drei ein halb"),
])
def test_flat_raw_fields(name, written, spoken):
    rule = numeric_rules(True)[name]
    fields = rewrite.one_top_rewrite(written, rule.fields_tagger)
    assert "{" not in fields and "preserve_order" not in fields
    assert rewrite.one_top_rewrite(fields, rule.fields_verbalizer) == spoken
    tagged = name + " { " + fields + " }"
    assert rule.verbalize(tagged) == spoken


def test_fraction_mixed_optional_und_weight():
    rule = numeric_rules(False)["fraction"]
    candidates = rule.normalize("1 1/2", nbest=10)
    assert candidates[0] == "ein ein halb"
    assert "ein und ein halb" in candidates


def test_cardinal_upper_bound_and_compound_groups():
    rule = numeric_rules(True)["cardinal"]
    written = "999999999999999999999999"
    group = "neun hundert neun und neunzig"
    spoken = " ".join(group + " " + quantity for quantity in (
        "trilliarden", "trillionen", "billiarden", "billionen", "milliarden", "millionen", "tausend"))
    spoken += " " + group
    assert rule.normalize(written) == spoken
    assert pynini.compose("1" + "0" * 24, rule.graph).start() == pynini.NO_STATE_ID
    assert pynini.compose("01", rule.graph).start() == pynini.NO_STATE_ID
    assert rule.normalize("100000000000000000000000") == "ein hundert trilliarden"


@pytest.mark.parametrize("name", ["cardinal", "decimal", "ordinal", "fraction"])
def test_no_default_field_reordering(name):
    module = import_module(ROOT + name)
    rule = numeric_rules(True)[name]
    assert module.TOKEN_ORDER == {}
    assert rule.token_orders == {}


def test_shared_digit_graph_is_unchanged():
    before = byte.DIGIT.copy()
    numeric_rules(True)
    numeric_rules(False)
    assert pynini.equivalent(before, byte.DIGIT)
    assert pynini.compose("12", byte.DIGIT).start() == pynini.NO_STATE_ID


@pytest.mark.parametrize("name,written", [
    ("decimal", "1000 millionen"),
    ("decimal", "1,"),
    ("cardinal", '1"'),
    ("cardinal", "1\\"),
    ("ordinal", "1"),
    ("fraction", "1/"),
])
def test_unsupported_numeric_inputs_are_rejected(name, written):
    assert pynini.compose(pynini.accep(pynini.escape(written)), numeric_rules(True)[name].tagger).start() == pynini.NO_STATE_ID
