from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import pynutil, rewrite

from tn.processor import Processor
from tn.utils import get_abs_path, load_labels


ROOT = "itn.german.rules."
DATA = Path(__file__).parent / "data"
CONTEXT_CASES = {"einhundert cent", "zweidollarzwanzig", "zwei pfund und ein hundert penny"}


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream]


@pytest.mark.parametrize("name", ["money", "measure"])
def test_rule_modules_exist(name):
    assert util.find_spec(ROOT + name) is not None
    assert getattr(import_module(ROOT + name), name.title()) is not None


@pytest.fixture(scope="module")
def rules():
    from itn.german.rules.cardinal import Cardinal
    from itn.german.rules.decimal import Decimal
    from itn.german.rules.fraction import Fraction
    from itn.german.rules.measure import Measure
    from itn.german.rules.money import Money
    from tn.german.rules.cardinal import Cardinal as TNCardinal
    from tn.german.rules.decimal import Decimal as TNDecimal
    from tn.german.rules.fraction import Fraction as TNFraction
    from tn.german.rules.ordinal import Ordinal as TNOrdinal

    tn_cardinal = TNCardinal(deterministic=False)
    tn_ordinal = TNOrdinal(tn_cardinal, deterministic=False)
    cardinal = Cardinal(tn_cardinal)
    decimal = Decimal(cardinal, TNDecimal(tn_cardinal, deterministic=False))
    fraction = Fraction(cardinal, TNFraction(tn_cardinal, tn_ordinal, deterministic=False))
    return {
        "cardinal": cardinal,
        "decimal": decimal,
        "fraction": fraction,
        "money": Money(cardinal, decimal),
        "measure": Measure(cardinal, decimal, fraction),
    }


def accepts(graph, text):
    return (pynini.accep(pynini.escape(text)) @ graph).start() != pynini.NO_STATE_ID


@pytest.fixture(scope="module")
def sentence_context(rules):
    rules = {name: rules[name] for name in ("cardinal", "money", "measure")}
    orders = {}
    for rule in rules.values():
        orders.update(rule.token_orders)
    word = Processor("word", ordertype="itn", token_orders=orders)
    word.tagger = word.add_tokens(word.tag_field("value", word.NOT_SPACE.plus))
    word.build_verbalizer()
    sentence = Processor("financial_context", ordertype="itn", token_orders=orders)
    token = pynini.union(
        *[pynutil.add_weight(rule.tagger, 1.1) for rule in rules.values()],
        pynutil.add_weight(word.tagger, 100),
    )
    sentence.tagger = (token + pynini.closure(pynutil.delete(" ") + token)).optimize()
    verbalizer = pynini.union(*[rule.verbalizer for rule in rules.values()], word.verbalizer)
    sentence.verbalizer = (
        verbalizer + pynini.closure(pynini.accep(" ") + verbalizer)
    ).optimize()
    return sentence


@pytest.mark.parametrize("spoken,written", source_pairs("money"))
def test_source_money(rules, sentence_context, spoken, written):
    rule = rules["money"]
    if spoken in CONTEXT_CASES:
        assert not accepts(rule.tagger, spoken)
        assert sentence_context.normalize(spoken) == written
        result = sentence_context.normalize_with_mapping(spoken)
        assert result.output_text == written
        for mapping in result.mappings:
            assert mapping.input_text == spoken[mapping.input_start:mapping.input_end]
            assert mapping.output_text == written[mapping.output_start:mapping.output_end]
        return
    assert rule.normalize(spoken) == written
    ordered = rule.token_parser().reorder(rule.tag(spoken))
    assert rewrite.one_top_rewrite(pynini.escape(ordered), rule.verbalizer) == written


@pytest.mark.parametrize("spoken,written", source_pairs("measure"))
def test_source_measure(rules, spoken, written):
    rule = rules["measure"]
    assert rule.normalize(spoken) == written
    ordered = rule.token_parser().reorder(rule.tag(spoken))
    assert rewrite.one_top_rewrite(pynini.escape(ordered), rule.verbalizer) == written


@pytest.mark.parametrize("name,spoken,written", [
    ("money", "null euro", "€0"),
    ("money", "eins euro", "€1"),
    ("money", "null cent", "€0,00"),
    ("money", "null euro und null cent", "€0,00"),
    ("money", "ein dollar und ein penny", "$1,01"),
    ("money", "zwei euro neunundneunzig", "€2,99"),
    ("money", "zwanzigcent", "€0,20"),
    ("money", "zwei\tus dollar\nvier", "$2,04"),
    ("money", "eine millionen euro", "€1 millionen"),
    ("measure", "null stunden", "0 h"),
    ("measure", "eins stunden", "1 h"),
    ("measure", "null kilometer pro stunde", "0 km/h"),
    ("measure", "null komma null kilometer", "0,0 km"),
    ("measure", "minus\telf\nkilogramm", "-11 kg"),
    ("measure", "minus drei und ein halb fuß", "-3 1/2 ft"),
    ("measure", "zwei kilometerpro stunde", "2 km/h"),
    ("measure", "zwei prosekunde", "2 /s"),
    ("measure", "zwei meilen", "2 mi"),
    ("measure", "zwei grad celsius", "2 °C"),
    ("measure", "zwei quadrat meter", "2 m²"),
])
def test_edge_readings(rules, name, spoken, written):
    assert rules[name].normalize(spoken) == written


@pytest.mark.parametrize("name,spoken,written,members", [
    ("money", "eine euro", "€1", {"integer_part": "eine", "currency": "euro"}),
    ("money", "zwei euro und ein penny", "€2,01",
     {"integer_part": "zwei", "currency": "euro", "conjunction": "und", "minor_part": "ein", "currency_minor": "penny"}),
    ("money", "ein cent", "€0,01", {"minor_part": "ein", "currency_minor": "cent"}),
    ("money", "null penny", "€0,00", {"minor_part": "null", "currency_minor": "penny"}),
    ("money", "zwei euro zwei", "€2,02", {"integer_part": "zwei", "currency": "euro", "minor_part": "zwei"}),
    ("money", "zwei komma zwei euro", "€2,2",
     {"integer_part": "zwei", "fractional_part": "zwei", "currency": "euro"}),
    ("money", "eine million us dollar", "$1 million",
     {"integer_part": "eine", "quantity": "million", "currency": "us dollar"}),
    ("money", "zwei komma zwei null null millionen euro", "€2,200 millionen",
     {"integer_part": "zwei", "fractional_part": "zwei null null", "quantity": "millionen", "currency": "euro"}),
    ("measure", "eine stunden", "1 h", {"integer": "eine", "units": "stunden"}),
    ("measure", "minus elf kilogramm", "-11 kg", {"negative": "minus", "integer": "elf", "units": "kilogramm"}),
    ("measure", "zwei kilometer pro stunde", "2 km/h", {"integer": "zwei", "units": "kilometer pro stunde"}),
    ("measure", "eins komma null zwei pro quadrat meter", "1,02 /m²",
     {"integer_part": "eins", "fractional_part": "null zwei", "units": "pro quadrat meter"}),
    ("measure", "eine million stunden", "1 million h",
     {"integer_part": "eine", "quantity": "million", "units": "stunden"}),
    ("measure", "ein ein halb fuß", "1 1/2 ft",
     {"integer_part": "ein", "numerator": "ein", "denominator": "halb", "units": "fuß"}),
    ("measure", "drei und ein halb fuß", "3 1/2 ft",
     {"integer_part": "drei", "conjunction": "und", "numerator": "ein", "denominator": "halb", "units": "fuß"}),
    ("measure", "minus minus ein halb fuß", "--1/2 ft",
     {"negative": "minus", "fraction_negative": "minus", "numerator": "ein", "denominator": "halb", "units": "fuß"}),
])
def test_raw_mapping_and_reordered_verbalization(rules, name, spoken, written, members):
    rule = rules[name]
    tagged = rule.tag(spoken)
    parser = rule.token_parser()
    ordered = parser.reorder(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert ordered == parser.reorder(ordered)
    assert rewrite.one_top_rewrite(pynini.escape(ordered), rule.verbalizer) == written
    assert rule.verbalize(tagged) == written
    assert rule.normalize(spoken) == written
    candidates = rule.normalize(spoken, nbest=10)
    results = rule.normalize_with_mapping(spoken, nbest=10)
    assert [result.output_text for result in results] == candidates
    for result in results:
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.token_type == name
        assert mapping.kind == "replace"
        assert (mapping.input_start, mapping.input_end) == (0, len(spoken))
        assert (mapping.output_start, mapping.output_end) == (0, len(result.output_text))
        assert mapping.input_text == spoken
        assert mapping.output_text == result.output_text


@pytest.mark.parametrize("name,spoken", [
    ("money", "zwei quadrat euro"),
    ("money", 'zwei eu"ro'),
    ("money", "zwei eu\\ro"),
    ("money", "zwei komma ein euro"),
    ("money", "ein komma zwei euro"),
    ("money", "minus zwei euro"),
    ("money", "hundert euro hundert"),
    ("money", "zweiundzwanzig Euro"),
    ("measure", 'zwei me"ter'),
    ("measure", "zwei me\\ter"),
    ("measure", "zwei pro stunden"),
    ("measure", "drei halb meter"),
    ("measure", "ein halbe meter"),
    ("measure", "zwei Meter"),
    ("measure", "acht a"),
    ("measure", "a acht"),
])
def test_rejected_inputs(rules, name, spoken):
    assert not accepts(rules[name].tagger, spoken)


@pytest.mark.parametrize("name,table,spoken_word,written_symbol", [
    ("money", "currency.tsv", 'eu"ro\\\\\\[x]', "€"),
    ("measure", "measurements.tsv", 'me"ter\\\\\\[x]', 'u"\\x'),
])
def test_escaped_raw_fields(rules, monkeypatch, name, table, spoken_word, written_symbol):
    string_file = pynini.string_file

    def labels(path):
        graph = string_file(path)
        if Path(path).name == table:
            graph |= pynini.cross(pynini.escape(written_symbol), pynini.escape(spoken_word))
        return graph

    monkeypatch.setattr(pynini, "string_file", labels)
    module = import_module(ROOT + name)
    if name == "money":
        rule = module.Money(rules["cardinal"], rules["decimal"])
        field = "currency"
        written = written_symbol + "2"
    else:
        rule = module.Measure(rules["cardinal"], rules["decimal"], rules["fraction"])
        field = "units"
        written = "2 " + written_symbol
    spoken = "zwei " + spoken_word
    tagged = rule.tag(spoken)
    parser = rule.token_parser()
    ordered = parser.reorder(tagged)
    assert parser.tokens[0].members[field] == spoken_word
    assert '\\"' in tagged and "\\\\" in tagged
    assert ordered == parser.reorder(ordered)
    assert rewrite.one_top_rewrite(pynini.escape(ordered), rule.verbalizer) == written
    result = rule.normalize_with_mapping(spoken)
    assert result.output_text == written
    assert result.mappings[0].input_text == spoken
    assert result.mappings[0].output_text == written


@pytest.mark.parametrize("name", ["money", "measure"])
def test_order_and_empty_api(rules, name):
    rule = rules[name]
    assert rule.ordertype == "itn"
    assert rule.token_orders == import_module(ROOT + name).TOKEN_ORDER
    assert rule.normalize("") == rule.tag("") == rule.verbalize("") == ""
    assert rule.normalize("", nbest=2) == [""]
    assert rule.normalize_with_mapping("").mappings == ()
    for invalid in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            rule.normalize("zwei euro", nbest=invalid)


def test_all_candidate_mappings(rules):
    for spoken in ("zwei quadrat meter", "zwei mikrogramm", "zwei hundert drei tausend vier fünftel meter"):
        rule = rules["measure"]
        candidates = rule.normalize(spoken, nbest=10)
        assert len(candidates) > 1
        results = rule.normalize_with_mapping(spoken, nbest=10)
        assert [result.output_text for result in results] == candidates
        for result in results:
            assert result.mappings[0].input_text == spoken
            assert result.mappings[0].output_text == result.output_text


@pytest.mark.parametrize("row", load_labels(get_abs_path("german/data/money/currency.tsv")))
def test_all_major_currency_rows(rules, row):
    symbol, word = row
    spoken = "eine " + word
    rule = rules["money"]
    assert rule.normalize(spoken) == symbol + "1"
    parser = rule.token_parser()
    parser.parse(rule.tag(spoken))
    assert parser.tokens[0].members == {"integer_part": "eine", "currency": word}


@pytest.mark.parametrize("row", load_labels(get_abs_path("german/data/measure/measurements.tsv")))
def test_all_measurement_rows(rules, row):
    symbol, word = row[:2]
    rule = rules["measure"]
    assert "2 " + symbol in rule.normalize("zwei " + word, nbest=20)
    assert "2 /" + symbol in rule.normalize("zwei pro " + word, nbest=20)
