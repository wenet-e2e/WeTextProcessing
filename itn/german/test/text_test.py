from importlib import import_module, util
from pathlib import Path

import pynini
import pytest
from pynini.lib import pynutil, rewrite

from tn.processor import Processor, RuleSpec
from tn.token_parser import escape_value


ROOT = "itn.german.rules."
DATA = Path(__file__).parent / "data"
NAMES = ("electronic", "telephone", "whitelist", "word", "punctuation")
MARKS = "!#$%&'()*+,-./:;<=>?@^_`{|}~"


def source_pairs(category):
    with (DATA / ("test_cases_" + category + ".txt")).open(encoding="utf-8") as stream:
        return [tuple(line.rstrip("\r\n").split("~")) for line in stream if "~" in line]


def accepts(graph, text):
    return (pynini.accep(pynini.escape(text)) @ graph).start() != pynini.NO_STATE_ID


@pytest.fixture(scope="module")
def rules():
    modules = {}
    for name in NAMES:
        assert util.find_spec(ROOT + name) is not None, "missing German ITN rule: " + name
        modules[name] = import_module(ROOT + name)
    from itn.german.rules.cardinal import Cardinal
    from tn.german.rules.cardinal import Cardinal as TNCardinal
    from tn.german.rules.electronic import Electronic as TNElectronic
    from tn.german.rules.whitelist import WhiteList as TNWhiteList

    cardinal = TNCardinal(deterministic=False)
    punctuation = modules["punctuation"].Punctuation()
    return {
        "electronic": modules["electronic"].Electronic(TNElectronic(deterministic=False)),
        "telephone": modules["telephone"].Telephone(cardinal),
        "whitelist": modules["whitelist"].WhiteList(TNWhiteList(deterministic=False)),
        "punctuation": punctuation,
        "word": modules["word"].Word(punctuation),
        "cardinal": Cardinal(cardinal),
    }


class TextPipeline(Processor):
    def __init__(self, rules):
        orders = {}
        for rule in rules.values():
            orders.update(rule.token_orders)
        super().__init__("german_text_test", ordertype="itn", token_orders=orders)
        self.rules = rules
        self.build_fst("german_text_test", cache_dir=False, overwrite_cache=False)

    def build_tagger_and_verbalizer(self):
        specs = [RuleSpec(self.rules[name], 100 if name == "word" else 1.0 if name == "whitelist" else 1.1)
                 for name in ("whitelist", "electronic", "telephone", "cardinal", "word")]
        punctuation = self.rules["punctuation"]
        punct = pynutil.add_weight(punctuation.tagger, 1.1)
        token = punct.star + self.tagger_union(specs) + punct.star
        self.tagger = (
            self.DELETE_SPACE + token + (self.DELETE_EXTRA_SPACE + token).star + self.DELETE_SPACE
        ).optimize()
        verbalizer = self.verbalizer_union(specs + [RuleSpec(punctuation)])
        self.verbalizer = (verbalizer + (pynini.accep(" ") + verbalizer).star).optimize()


@pytest.fixture(scope="module")
def pipeline(rules):
    return TextPipeline(rules)


@pytest.mark.parametrize("name", NAMES)
def test_rule_modules_exist(name):
    assert util.find_spec(ROOT + name) is not None, "missing German ITN rule: " + name


@pytest.mark.parametrize("category,count", [("electronic", 9), ("telephone", 1), ("whitelist", 6), ("word", 49)])
def test_source_fixture_counts(category, count):
    assert len(source_pairs(category)) == count


@pytest.mark.parametrize("spoken,written", source_pairs("electronic"))
def test_source_electronic(rules, spoken, written):
    assert rules["electronic"].normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", [
    ("a b c punkt com punkt", "abc.com."),
    ("a b c at g mail punkt com punkt", "abc@gmail.com."),
    ("h t t p doppelpunkt slash slash a b c punkt com punkt", "http://abc.com."),
])
def test_electronic_spoken_terminal_period(rules, pipeline, spoken, written):
    assert rules["electronic"].normalize(spoken) == written
    assert pipeline.normalize(spoken) == written
    mapped = pipeline.normalize_with_mapping(spoken)
    assert mapped.output_text == written
    assert mapped.mappings[0].input_text == spoken


@pytest.mark.parametrize("spoken,written", source_pairs("telephone"))
def test_source_telephone(rules, spoken, written):
    assert rules["telephone"].normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", source_pairs("whitelist"))
def test_source_whitelist(pipeline, spoken, written):
    assert pipeline.normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", source_pairs("word"))
def test_source_word(pipeline, spoken, written):
    assert pipeline.normalize(spoken) == written


@pytest.mark.parametrize("name,spoken,written,members", [
    ("electronic", "a b c at g mail punkt com", "abc@gmail.com",
     {"username": "a b c", "domain": "g mail punkt com"}),
    ("electronic", "w w w punkt a b c punkt com", "www.abc.com",
     {"protocol": "w w w punkt", "domain": "a b c punkt com"}),
    ("electronic", "at a b c", "@abc", {"username": "a b c"}),
    ("electronic", "a b c punkt com", "abc.com", {"domain": "a b c punkt com"}),
    ("telephone", "null vier eins eins eins zwei drei vier eins zwei drei vier", "(0411) 1234-1234",
     {"country_code": "null vier eins eins", "number_part": "eins zwei drei vier eins zwei drei vier"}),
    ("whitelist", "zum beispiel", "z.B.", {"value": "zum beispiel"}),
    ("word", 'Äpfel"\\Baum', 'Äpfel"\\Baum', {"value": 'Äpfel"\\Baum'}),
    ("punctuation", "$", "$", {"value": "$"}),
])
def test_raw_fields_and_direct_mapping(rules, name, spoken, written, members):
    rule = rules[name]
    tagged = rule.tag(spoken)
    parser = rule.token_parser()
    reordered = parser.reorder(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].members == members
    assert parser.tokens[0].order == list(members)
    assert rewrite.one_top_rewrite(pynini.escape(reordered), rule.verbalizer) == written
    assert rule.verbalize(tagged) == written
    assert rule.normalize(spoken) == written
    results = rule.normalize_with_mapping(spoken, nbest=5, include_identity=True)
    assert [result.output_text for result in results] == rule.normalize(spoken, nbest=5)
    for result in results:
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.token_type == name
        assert mapping.input_text == spoken
        assert mapping.output_text == result.output_text
        assert (mapping.input_start, mapping.input_end) == (0, len(spoken))
        assert (mapping.output_start, mapping.output_end) == (0, len(result.output_text))


@pytest.mark.parametrize("name,attribute,spoken,written", [
    ("electronic", "username_graph", "a eins bindestrich b", "a1-b"),
    ("electronic", "domain_graph", "g mail punkt com", "gmail.com"),
    ("electronic", "protocol_graph", "h t t p s doppelpunkt slash slash w w w punkt", "https://www."),
    ("telephone", "digits", "eins", "1"),
    ("telephone", "country_code_graph", "null vier eins eins", "(0411)"),
    ("telephone", "number_part_graph", "eins zwei drei vier eins zwei drei vier", "1234-1234"),
    ("whitelist", "graph", "doktor", "Dr."),
    ("whitelist", "graph", "kilogramm", "kg"),
])
def test_public_semantic_graphs(rules, name, attribute, spoken, written):
    assert written in rewrite.top_rewrites(spoken, getattr(rules[name], attribute), 10)


@pytest.mark.parametrize("name", NAMES)
def test_order_contract(rules, name):
    rule = rules[name]
    assert rule.ordertype == "itn"
    assert rule.token_orders == import_module(ROOT + name).TOKEN_ORDER


@pytest.mark.parametrize("name", NAMES)
def test_no_empty_tagger_path_and_empty_api(rules, name):
    rule = rules[name]
    assert not accepts(rule.tagger, "")
    assert rule.tag("") == ""
    assert rule.verbalize("") == ""
    assert rule.normalize("") == ""
    assert rule.normalize("", nbest=5) == [""]
    assert rule.normalize_with_mapping("").mappings == ()


@pytest.mark.parametrize("text", [
    '"', "\\", '\\"', 'a"b', r"a\b", r"a\\b", r"[abc]", "Grüße", "MiXeD", "—", "yahoo!", "a'b", "\v", "\f",
])
def test_word_safe_roundtrip(rules, text):
    rule = rules["word"]
    tagged = rule.tag(text)
    assert 'value: "' + escape_value(text) + '"' in tagged
    parser = rule.token_parser()
    reordered = parser.reorder(tagged)
    assert parser.tokens[0].members == {"value": text}
    assert rewrite.one_top_rewrite(pynini.escape(reordered), rule.verbalizer) == text
    assert rule.normalize_with_mapping(text, include_identity=True).mappings[0].input_text == text


@pytest.mark.parametrize("mark", list(MARKS))
def test_source_punctuation_marks(rules, mark):
    punctuation = rules["punctuation"]
    assert accepts(punctuation.tagger, mark)
    assert not accepts(rules["word"].tagger, mark)
    assert punctuation.normalize(mark) == mark
    tagged = punctuation.tag(mark)
    reordered = punctuation.token_parser().reorder(tagged)
    assert rewrite.one_top_rewrite(pynini.escape(reordered), punctuation.verbalizer) == mark


@pytest.mark.parametrize("text", ['"', "\\", "[", "]", "—", "!!", "<b>"])
def test_punctuation_source_exclusions(rules, text):
    assert not accepts(rules["punctuation"].tagger, text)
    assert rules["word"].normalize(text) == text


@pytest.mark.parametrize("space", [" ", "  ", "\t", "\n", "\r", "\u00a0", " \t\n\r\u00a0 "])
def test_word_pure_whitespace_verbalizer(rules, space):
    word = rules["word"]
    assert not accepts(word.tagger, space)
    tagged = 'word { value: "' + escape_value(space) + '" }'
    reordered = word.token_parser().reorder(tagged)
    assert rewrite.one_top_rewrite(pynini.escape(reordered), word.verbalizer) == ""
    assert word.verbalize(tagged) == ""


def test_word_nonbreaking_space_verbalizer(rules):
    assert rules["word"].verbalize('word { value: "a\u00a0b" }') == "a b"


@pytest.mark.parametrize("name,spoken", [
    ("word", "zwei wörter"),
    ("telephone", "null vier eins eins eins zwei drei vier eins zwei drei null"),
    ("telephone", "vier eins eins eins zwei drei vier eins zwei drei vier"),
    ("telephone", "null vier eins eins eins zwei drei vier eins zwei drei"),
    ("telephone", "null vier eins eins ein zwei drei vier eins zwei drei vier"),
    ("telephone", "null vier eins eins eine zwei drei vier eins zwei drei vier"),
    ("telephone", "null vier eins eins einer zwei drei vier eins zwei drei vier"),
    ("electronic", "a b c at com"),
    ("whitelist", "Doktor"),
])
def test_invalid_inputs(rules, name, spoken):
    assert not accepts(rules[name].tagger, spoken)


@pytest.mark.parametrize("length", [1, 257, 1025])
def test_electronic_unbounded_field_reordering(rules, length):
    rule = rules["electronic"]
    username = " ".join("a" * length)
    domain = " ".join("b" * length) + " punkt com"
    email = 'electronic { domain: "' + domain + '" username: "' + username + '" }'
    url = 'electronic { domain: "' + domain + '" protocol: "w w w punkt" }'
    assert rule.verbalize(email) == "a" * length + "@" + "b" * length + ".com"
    assert rule.verbalize(url) == "www." + "b" * length + ".com"
    assert rule.normalize(username + " at " + domain) == "a" * length + "@" + "b" * length + ".com"


@pytest.mark.parametrize("spoken,written", [
    ("at a b c punkt", "@abc."),
    ("h t t p doppelpunkt slash slash a b c punkt com", "http://abc.com"),
])
def test_electronic_social_and_url(rules, spoken, written):
    assert rules["electronic"].normalize(spoken) == written


def test_electronic_rejects_written_terminal_period(rules):
    assert not accepts(rules["electronic"].tagger, "a b c punkt com.")


def test_fixture_spaces_are_retained():
    assert ("x ", "x") in source_pairs("word")
    assert ("", "") in source_pairs("word")


def test_pipeline_uses_merged_schema_and_keeps_case(pipeline):
    assert pipeline.normalize("MiXeD doktor Dao") == "MiXeD Dr. Dao"
    assert pipeline.normalize("w w w punkt a b c punkt com") == "www.abc.com"
    assert not accepts(pipeline.tagger, "")
    assert not accepts(pipeline.tagger, " \t\n")
