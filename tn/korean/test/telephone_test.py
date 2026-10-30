import pynini
import pytest

from tn.korean.rules.telephone import Telephone


@pytest.fixture(scope="module")
def rule():
    return Telephone()


@pytest.mark.parametrize("written, members, spoken", [
    ("010-3713-7050", {"number_part": "010-3713-7050"}, "영일영 삼칠일삼 칠영오영"),
    ("+82 010-1234-5678", {"country_code": "+82", "number_part": "010-1234-5678"},
     "국가번호 팔이 영일영 일이삼사 오육칠팔"),
    ("+1-123.456.7890", {"country_code": "+1", "number_part": "123.456.7890"},
     "국가번호 일 일이삼 사오육 칠팔구영"),
    ("(02) 1234-5678", {"number_part": "(02) 1234-5678"}, "영이 일이삼사 오육칠팔"),
    ("000.000.0000", {"number_part": "000.000.0000"}, "영영영 영영영 영영영영"),
    ("010 1234 5678", {"number_part": "010 1234 5678"}, "영일영 일이삼사 오육칠팔"),
])
def test_raw_fields_and_mapping(rule, written, members, spoken):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert parser.tokens[0].members == members
    assert rule.normalize(written) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert result.mappings[0].input_text == written


@pytest.mark.parametrize("raw", ['a"b', r"a\b", 'a"\\b'])
def test_quoted_field_round_trip(rule, raw):
    # These characters are not telephone syntax, but the codec must decode them.
    tagged = pynini.shortestpath(pynini.escape(raw) @ rule.tag_field("value", rule.VSIGMA)).string()
    parser = rule.token_parser()
    parser.parse("telephone { " + tagged + " }")
    assert parser.tokens[0].members == {"value": raw}
    decoded = pynini.shortestpath(pynini.escape(tagged) @ rule.verbalize_field("value")).string()
    assert decoded == raw
