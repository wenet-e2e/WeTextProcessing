import pynini
import pytest

from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.time import TOKEN_ORDER, Time


@pytest.fixture(scope="module")
def rule():
    return Time(Cardinal())


@pytest.mark.parametrize("written, members, spoken", [
    ("오전10시30분", {"suffix": "오전", "hours": "10시", "minutes": "30분"}, "오전 열시 삼십분"),
    ("00:30:00", {"hours": "00", "minutes": ":30", "seconds": ":00"}, "영시 삼십분"),
    ("23:00", {"hours": "23", "minutes": ":00"}, "이십삼시"),
    ("24:03:38", {"hours": "24", "minutes": ":03", "seconds": ":38"}, "이십사시 삼분 삼십팔초"),
    ("오후 3시 반", {"suffix": "오후", "hours": "3시", "minutes": "반"}, "오후 세시 반"),
    ("정오", {"hours": "정오"}, "열두시"),
    ("자정", {"hours": "자정"}, "영시"),
    ("10:30 오후", {"hours": "10", "minutes": ":30", "suffix": "오후"}, "오후 열시 삼십분"),
    ("3시 0분 오전", {"hours": "3시", "minutes": "0분", "suffix": "오전"}, "오전 세시 "),
])
def test_raw_fields_and_mapping(rule, written, members, spoken):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert parser.tokens[0].members == members
    assert rule.normalize(written) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert result.mappings[0].input_text == written


def test_postfix_order(rule):
    assert TOKEN_ORDER == {"time": ["suffix", "hours", "minutes", "seconds"]}
    parser = rule.token_parser()
    parser.parse(rule.tag("10:30 오후"))
    assert parser.tokens[0].order == ["hours", "minutes", "suffix"]
    parser.parse(parser.reorder(rule.tag("10:30 오후")))
    assert parser.tokens[0].order == TOKEN_ORDER["time"][:3]


@pytest.mark.parametrize("raw", ['a"b', r"a\b", 'a"\\b'])
def test_quoted_field_round_trip(rule, raw):
    # Time syntax cannot contain these characters; exercise its wire codec.
    tagged = pynini.shortestpath(pynini.escape(raw) @ rule.tag_field("value", rule.VSIGMA)).string()
    parser = rule.token_parser()
    parser.parse("time { " + tagged + " }")
    assert parser.tokens[0].members == {"value": raw}
    decoded = pynini.shortestpath(pynini.escape(tagged) @ rule.verbalize_field("value")).string()
    assert decoded == raw
