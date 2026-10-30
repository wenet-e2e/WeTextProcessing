from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import byte, rewrite

from tn.token_parser import TokenParser


ROOT = "itn.german.rules."
DATA = Path(__file__).parent / "data"


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream if "~" in line]


@pytest.fixture(scope="module")
def rules():
    modules = {}
    for name in ("cardinal", "decimal", "ordinal", "fraction"):
        assert util.find_spec(ROOT + name) is not None, "missing German ITN rule: " + name
        modules[name] = import_module(ROOT + name)
    from tn.german.rules.cardinal import Cardinal as TNCardinal
    from tn.german.rules.decimal import Decimal as TNDecimal
    from tn.german.rules.fraction import Fraction as TNFraction
    from tn.german.rules.ordinal import Ordinal as TNOrdinal

    tn_cardinal = TNCardinal(deterministic=False)
    tn_ordinal = TNOrdinal(tn_cardinal, deterministic=False)
    cardinal = modules["cardinal"].Cardinal(tn_cardinal)
    return {
        "cardinal": cardinal,
        "ordinal": modules["ordinal"].Ordinal(cardinal, tn_ordinal),
        "decimal": modules["decimal"].Decimal(cardinal, TNDecimal(tn_cardinal, deterministic=False)),
        "fraction": modules["fraction"].Fraction(
            cardinal, TNFraction(tn_cardinal, tn_ordinal, deterministic=False),
        ),
    }


def accepts(graph, text):
    return pynini.compose(pynini.accep(pynini.escape(text)), graph).start() != pynini.NO_STATE_ID


@pytest.mark.parametrize("name", ["cardinal", "decimal", "ordinal", "fraction"])
def test_rule_modules_exist(name):
    assert util.find_spec(ROOT + name) is not None, "missing German ITN rule: " + name


@pytest.mark.parametrize("name,spoken,written,members", [
    ("cardinal", "minus dreiundzwanzig", "-23", {"negative": "minus", "integer": "dreiundzwanzig"}),
    ("cardinal", "ein hundert drei und zwanzig", "123", {"integer": "ein hundert drei und zwanzig"}),
    ("cardinal", "fünfundzwanzig", "25", {"integer": "fünfundzwanzig"}),
    ("ordinal", "fünfundzwanzigtausendeinhundertelftem", "25111.",
     {"integer": "fünfundzwanzigtausendeinhundertelftem"}),
    ("decimal", "minus sechzig komma zwei vier null null", "-60,2400",
     {"negative": "minus", "integer_part": "sechzig", "fractional_part": "zwei vier null null"}),
    ("decimal", "eine million", "1 million", {"integer_part": "eine", "quantity": "million"}),
    ("fraction", "ein halb", "1/2", {"numerator": "ein", "denominator": "halb"}),
    ("fraction", "ein ein halb", "1 1/2",
     {"integer_part": "ein", "numerator": "ein", "denominator": "halb"}),
    ("fraction", "minus drei und ein halb", "-3 1/2",
     {"negative": "minus", "integer_part": "drei", "conjunction": "und", "numerator": "ein", "denominator": "halb"}),
])
def test_raw_fields_mapping_and_nbest(rules, name, spoken, written, members):
    rule = rules[name]
    tagged = rule.tag(spoken)
    parser = TokenParser(rule.token_orders)
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert rule.normalize(spoken) == written
    assert rule.verbalize(tagged) == written
    candidates = rule.normalize(spoken, nbest=5)
    assert candidates[0] == written
    results = rule.normalize_with_mapping(spoken, nbest=5)
    assert [result.output_text for result in results] == candidates
    for result in results:
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.input_text == spoken
        assert mapping.output_text == result.output_text
        assert (mapping.input_start, mapping.input_end) == (0, len(spoken))
        assert (mapping.output_start, mapping.output_end) == (0, len(result.output_text))


@pytest.mark.parametrize("spoken,written", source_pairs("cardinal"))
def test_source_cardinal(rules, spoken, written):
    if spoken == written:
        assert not accepts(rules["cardinal"].tagger, spoken)
    else:
        assert rules["cardinal"].normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", source_pairs("decimal"))
def test_source_decimal(rules, spoken, written):
    assert rules["decimal"].normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", source_pairs("ordinal"))
def test_source_ordinal(rules, spoken, written):
    if spoken == written or spoken.startswith("dem "):
        # The sentence fixture belongs to the pipeline; no numeric rule consumes 'dem'.
        assert not accepts(rules["ordinal"].tagger, spoken)
    else:
        assert rules["ordinal"].normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", source_pairs("fraction"))
def test_source_fraction(rules, spoken, written):
    assert rules["fraction"].normalize(spoken) == written


@pytest.mark.parametrize("name,attribute,spoken,written", [
    ("cardinal", "graph_no_exception", "null", "0"),
    ("cardinal", "graph_no_exception", "eins", "1"),
    ("cardinal", "graph_no_exception", "zweihundertvierundfünfzig", "254"),
    ("cardinal", "digits", "ein", "1"),
    ("cardinal", "digits", "neun", "9"),
    ("cardinal", "hundreds", "neunhundertneunundneunzig", "999"),
    ("cardinal", "two_digits", "einundzwanzig", "21"),
    ("cardinal", "sign", "minus", "-"),
    ("decimal", "fractional_graph", "null eins null", "010"),
    ("decimal", "quantity_graph", "millionen", "millionen"),
    ("decimal", "quantity_graph", "milliarden", "milliarden"),
])
def test_public_semantic_graphs(rules, name, attribute, spoken, written):
    assert written in rewrite.top_rewrites(spoken, getattr(rules[name], attribute), 10)


@pytest.mark.parametrize("name,spoken,written", [
    ("decimal", "minus eins komma null fünf millionen", "-1,05 millionen"),
    ("fraction", "minus drei ein halb", "-3 1/2"),
    ("fraction", "minus drei und ein halb", "-3 1/2"),
])
def test_flat_raw_fields(rules, name, spoken, written):
    rule = rules[name]
    fields = rewrite.one_top_rewrite(spoken, rule.fields_tagger)
    assert "{" not in fields and "preserve_order" not in fields
    assert 'negative: "minus"' in fields
    assert rewrite.one_top_rewrite(fields, rule.fields_verbalizer) == written
    assert rule.verbalize(name + " { " + fields + " }") == written


@pytest.mark.parametrize("name,spoken,written", [
    ("decimal", "eine billion", "1 billion"),
    ("decimal", "einundzwanzig komma eins null", "21,10"),
    ("decimal", "neunhundertneunundneunzig millionen", "999 millionen"),
    ("fraction", "minus null nulltel", "-0/0"),
    ("fraction", "drei zwei ein hundertstel", "3 2/100"),
    ("fraction", "ein tausend eins zwei ein hundertstel", "1001 2/100"),
    ("fraction", "ein tausend eins und zwei ein hundertstel", "1001 2/100"),
    ("fraction", "ein einhundertstel", "1/100"),
])
def test_contextual_numbers(rules, name, spoken, written):
    assert rules[name].normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["null", "eins", "ein", "eine", "einer", "zwei", "drei", "vier", "fünf", "neun"])
def test_cardinal_single_digit_exceptions(rules, spoken):
    assert not accepts(rules["cardinal"].tagger, spoken)


@pytest.mark.parametrize("ending", ["ten", "tem", "ter", "tes", "te"])
def test_ordinal_inflections(rules, ending):
    assert rules["ordinal"].normalize("einundzwanzigs" + ending) == "21."
    assert not accepts(rules["ordinal"].tagger, "ers" + ending)


@pytest.mark.parametrize("name,spoken", [
    ("cardinal", "d r e i ß i g"),
    ("cardinal", "Dreißig"),
    ("cardinal", " minus dreißig"),
    ("cardinal", "min usdreißig"),
    ("decimal", "tausend millionen"),
    ("decimal", "eins komma"),
    ("decimal", "eins komma ein"),
    ("fraction", "vier halb"),
    ("fraction", "ein halbe"),
    ("fraction", "ein/halb"),
    ("ordinal", "zehn"),
])
def test_invalid_inputs_are_rejected(rules, name, spoken):
    assert not accepts(rules[name].tagger, spoken)


@pytest.mark.parametrize("name", ["cardinal", "decimal", "ordinal", "fraction"])
def test_order_contract(rules, name):
    assert import_module(ROOT + name).TOKEN_ORDER == {}
    assert rules[name].token_orders == {}
    assert rules[name].ordertype == "itn"


def test_shared_digit_graph_is_unchanged(rules):
    assert pynini.equivalent(byte.DIGIT, pynini.union(*"0123456789").optimize())
    assert not accepts(byte.DIGIT, "12")


def test_fraction_ambiguous_operands_keep_source_ranking_and_mapping(rules):
    rule = rules["fraction"]
    spoken = "zwei hundert drei tausend vier fünftel"
    candidates = rule.normalize(spoken, nbest=10)
    assert candidates[0] == "203004/5"
    assert "200 3004/5" in candidates
    results = rule.normalize_with_mapping(spoken, nbest=10)
    assert [result.output_text for result in results] == candidates
    for result in results:
        assert result.mappings[0].input_text == spoken
        assert result.mappings[0].output_text == result.output_text


def test_ordinal_does_not_double_cardinal_semantic_weights(rules):
    cardinal = pynini.shortestpath("ein hundert und drei und zwanzig" @ rules["cardinal"].graph).paths()
    ordinal = pynini.shortestpath("ein hundert und drei und zwanzigster" @ rules["ordinal"].graph).paths()
    assert float(ordinal.weight()) == pytest.approx(float(cardinal.weight()) + 0.01, abs=1e-7)
