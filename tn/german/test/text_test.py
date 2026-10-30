from functools import lru_cache
from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import byte, rewrite

from tn.token_parser import Token, TokenParser
from tn.utils import get_abs_path, load_labels


ROOT = "tn.german.rules."
DATA = Path(__file__).parent / "data"
NAMES = ("electronic", "telephone", "whitelist", "word", "punctuation")


@lru_cache(maxsize=None)
def text_rules(deterministic=True):
    modules = {}
    for name in NAMES:
        assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name
        modules[name] = import_module(ROOT + name)
    cardinal = import_module(ROOT + "cardinal").Cardinal(deterministic=deterministic)
    punctuation = modules["punctuation"].Punctuation(deterministic=deterministic)
    return {
        "electronic": modules["electronic"].Electronic(deterministic=deterministic),
        "telephone": modules["telephone"].Telephone(cardinal, deterministic=deterministic),
        "whitelist": modules["whitelist"].WhiteList(deterministic=deterministic),
        "word": modules["word"].Word(punctuation, deterministic=deterministic),
        "punctuation": punctuation,
    }


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8", newline="") as stream:
        return [tuple((line[:-1] if line.endswith("\n") else line).split("~"))
                for line in stream if "~" in line]


@pytest.mark.parametrize("name", NAMES)
def test_native_rule_exists(name):
    assert util.find_spec(ROOT + name) is not None, "missing German rule: " + name


@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("name,written,spoken,members", [
    ("electronic", "a1b2@gmail.com", "a eins b zwei at g mail punkt com",
     {"username": "a1b2", "domain": "gmail.com"}),
    ("electronic", "https://www.amazon.de.",
     "h t t p s doppelpunkt slash slash w w w punkt amazon punkt de.",
     {"protocol": "https://www.", "domain": "amazon.de."}),
    ("electronic", "@jensen.me", "at j e n s e n punkt m e", {"username": "jensen.me"}),
    ("electronic", "abc{d}.de", "a b c linke klammer auf d rechte klammer zu punkt de",
     {"domain": "abc{d}.de"}),
    ("telephone", "+49 0123-0000", "plus neun und vierzig null eins zwei drei null null null null",
     {"country_code": "+49", "number_part": "0123-0000"}),
    ("telephone", "(0411) 0234-1234", "null vier eins eins null zwei drei vier eins zwei drei vier",
     {"country_code": "(0411)", "number_part": "0234-1234"}),
    ("telephone", "0411 02341234", "null vier eins eins null zwei drei vier eins zwei drei vier",
     {"country_code": "0411", "number_part": "02341234"}),
    ("whitelist", "Dr.", "doktor", {"value": "Dr."}),
    ("whitelist", "2D-Mammogram", "zwei-D-Mammogram", {"value": "2D-Mammogram"}),
    ("word", 'grün\\pfad"中文', 'grün\\pfad"中文', {"value": 'grün\\pfad"中文'}),
    ("word", "yahoo!", "yahoo!", {"value": "yahoo!"}),
    ("punctuation", '"', '"', {"value": '"'}),
    ("punctuation", r"\[", r"\[", {"value": r"\["}),
    ("punctuation", "—…", "—…", {"value": "—…"}),
    ("punctuation", "<em>", "<em>", {"value": "<em>"}),
])
def test_raw_fields_semantics_and_mapping(deterministic, name, written, spoken, members):
    rule = text_rules(deterministic)[name]
    tagged = rule.tag(written)
    parser = TokenParser(rule.token_orders)
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert rule.normalize(written) == spoken
    assert rule.verbalize(tagged) == spoken
    result = rule.normalize_with_mapping(written, include_identity=True)
    assert result.output_text == spoken
    assert len(result.mappings) == 1
    mapping = result.mappings[0]
    assert mapping.input_text == written
    assert mapping.output_text == spoken
    assert (mapping.input_start, mapping.input_end) == (0, len(written))
    assert (mapping.output_start, mapping.output_end) == (0, len(spoken))


@pytest.mark.parametrize("spoken,written", source_pairs("electronic"))
def test_source_electronic(spoken, written):
    rules = text_rules()
    name = "electronic" if "." in written or "@" in written else "whitelist"
    assert rules[name].normalize(written) == spoken


@pytest.mark.parametrize("spoken,written", source_pairs("telephone"))
def test_source_telephone(spoken, written):
    assert text_rules()["telephone"].normalize(written) == spoken


@pytest.mark.parametrize("spoken,written", [
    (spoken, written) for spoken, written in source_pairs("word") if spoken == written
])
def test_source_identity_word_tokens(spoken, written):
    rules = text_rules()
    is_punct = pynini.compose(pynini.accep(pynini.escape(written)), rules["punctuation"].graph).start()
    rule = rules["punctuation"] if is_punct != pynini.NO_STATE_ID else rules["word"]
    assert rule.normalize(written) == spoken


@pytest.mark.parametrize("written,spoken", load_labels(get_abs_path("german/data/whitelist.tsv")))
def test_source_whitelist_map(written, spoken):
    rule = text_rules()["whitelist"]
    assert rule.normalize(written) == spoken
    assert rewrite.one_top_rewrite(written, rule.graph) == spoken


@pytest.mark.parametrize("name,attribute,written,spoken", [
    ("electronic", "username_graph", "a1-b", "a eins bindestrich b"),
    ("electronic", "domain_graph", "gmail.abc", "g mail punkt a b c"),
    ("electronic", "domain_graph", "amazon.com.de", "amazon punkt com punkt de"),
    ("electronic", "domain_graph", "Sinnesbereichen.de", "Sinnesbereichen punkt de"),
    ("electronic", "protocol_graph", "http://", "h t t p doppelpunkt slash slash"),
    ("telephone", "country_code_graph", "+49", "plus neun und vierzig"),
    ("telephone", "country_code_graph", "(012)", "null eins zwei"),
    ("telephone", "number_part_graph", "0001234", "null null null eins zwei drei vier"),
    ("whitelist", "graph", "z.B.", "zum beispiel"),
])
def test_public_bare_semantic_graphs(name, attribute, written, spoken):
    assert spoken in rewrite.top_rewrites(written, getattr(text_rules()[name], attribute), 20)


@pytest.mark.parametrize("name,written", [
    ("electronic", 'a"b@abc.de'), ("electronic", r"a\b@abc.de"),
    ("electronic", "ä@abc.de"), ("electronic", "abc"),
    ("telephone", "+1 1234567"), ("telephone", "+49 123456"),
    ("telephone", "+49 123-456-789"), ("telephone", "+49 1234\t1234"),
    ("whitelist", "dr."), ("whitelist", "DR."),
    ("word", "!"), ("word", "—"), ("word", "<em>"),
    ("word", "two words"), ("word", "a\u00a0b"), ("word", "a\tb"),
    ("word", "["), ("word", "]"), ("word", "\\"),
    ("punctuation", "! "), ("punctuation", "&"), ("punctuation", "§"), ("punctuation", "a!"),
])
def test_source_character_and_length_exclusions(name, written):
    assert pynini.compose(pynini.accep(pynini.escape(written)), text_rules()[name].tagger).start() == pynini.NO_STATE_ID


def test_punctuation_inventory_matches_source_exclusions():
    punctuation = text_rules()["punctuation"]
    for mark in ["!", '"', "—", "。", "[", "]", "\\"]:
        assert rewrite.one_top_rewrite(pynini.escape(mark), punctuation.punct) == mark
    for mark in ["&", "#", "@", "§", "™", "_"]:
        assert pynini.compose(pynini.accep(pynini.escape(mark)), punctuation.punct).start() == pynini.NO_STATE_ID


def test_whitelist_false_only_adds_source_alternatives():
    rules = text_rules(False)
    assert rules["whitelist"].normalize("dr.") == "doktor"
    assert rules["whitelist"].normalize("km") == "kilometer"
    assert set(rules["whitelist"].normalize("kb", nbest=10)) == {"kilobit", "kilobyte"}
    parser = TokenParser({})
    parser.parse(rules["whitelist"].tag("dr."))
    assert parser.tokens[0].members == {"value": "dr."}
    assert rules["whitelist"].normalize("Dr.") == "doktor"


def test_word_verbalizer_decodes_then_replaces_nbsp():
    token = Token("word")
    token.append("value", 'grün\u00a0"\\中文')
    assert text_rules()["word"].verbalize(token.string({})) == 'grün "\\中文'


@pytest.mark.parametrize("written", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_word_verbalizer_deletes_pure_whitespace(written):
    token = Token("word")
    token.append("value", written)
    assert text_rules()["word"].verbalize(token.string({})) == ""


def test_electronic_bare_terminal_period_not_postprocessed():
    rule = text_rules()["electronic"]
    assert rewrite.one_top_rewrite("abc.com.", rule.domain_graph) == "a b c punkt com punkt"
    assert rule.normalize("abc.com.") == "a b c punkt com."


def test_domain_known_and_spelled_candidates_keep_raw():
    rule = text_rules(False)["electronic"]
    candidates = rule.normalize("abc@gmail.com.", nbest=30)
    assert candidates[0] == "a b c at g mail punkt com."
    assert "a b c at g m a i l punkt c o m." in candidates
    parser = TokenParser({})
    parser.parse(rule.tag("abc@gmail.com."))
    assert parser.tokens[0].members == {"username": "abc", "domain": "gmail.com."}


@pytest.mark.parametrize("name", NAMES)
def test_explicit_empty_token_orders(name):
    assert import_module(ROOT + name).TOKEN_ORDER == {}
    assert text_rules()[name].token_orders == {}


def test_shared_byte_graphs_are_not_modified():
    space, digit = byte.SPACE.copy(), byte.DIGIT.copy()
    text_rules(True)
    text_rules(False)
    assert pynini.equivalent(space, byte.SPACE)
    assert pynini.equivalent(digit, byte.DIGIT)


def test_source_fixture_empty_and_leading_space_are_retained():
    pairs = source_pairs("word")
    assert pairs[0] == ("", "")
    assert ("x", " x") in pairs
    assert len(pairs) == 49


@pytest.mark.parametrize("country,spoken", [
    ("02", "null zwei"), ("01", "null eins"), ("00", "null null"),
    ("09", "null neun"), ("(012)", "null eins zwei"),
])
def test_telephone_leading_zero_roundtrip(country, spoken):
    rule = text_rules()["telephone"]
    written = country + " 2345678"
    expected = spoken + " zwei drei vier fünf sechs sieben acht"
    assert rule.normalize(written) == expected
    assert rule.verbalize(rule.tag(written)) == expected
    assert rule.normalize(written, nbest=10) == [expected]


def test_telephone_digits_do_not_use_counting_inflections():
    rule = text_rules(False)["telephone"]
    expected = "plus neun und vierzig eins zwei drei vier fünf sechs sieben"
    assert rule.normalize("+49 1234567", nbest=10) == [expected]
