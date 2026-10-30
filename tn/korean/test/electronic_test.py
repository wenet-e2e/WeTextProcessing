import pynini
import pytest

from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.electronic import Electronic


@pytest.fixture(scope="module")
def rule():
    return Electronic(Cardinal())


@pytest.mark.parametrize("written, members, spoken", [
    ("user01@gmail.co.kr", {"username": "user01@", "domain": "gmail.co.kr"},
     "user영일 골뱅이 gmail 닷 씨오 닷 케이알"),
    ("nvidia.co.kr", {"domain": "nvidia.co.kr"}, "nvidia 닷 씨오 닷 케이알"),
    ("사진.JPG", {"domain": "사진.JPG"}, "사진 닷 제이피지"),
    ("카드 마지막 4자리 7890", {"protocol": "카드 마지막 4자리", "domain": "7890"},
     "카드 마지막 네자리 칠팔구영"),
    ("1234-5678-9012-3456", {"kind": "credit_card", "card_number": "1234-5678-9012-3456"},
     "신용카드 일이삼사 오육칠팔 구영일이 삼사오육"),
    ("https://www.nvidia.com", {"protocol": "https://www.", "domain": "nvidia.com"},
     "HTTPS colon 슬래시 슬래시 WWW 점 nvidia 닷컴"),
    ("file:///a.com", {"protocol": "file:///", "domain": "a.com"},
     "file colon 슬래시 슬래시 슬래시 a 닷컴"),
])
def test_raw_fields_and_mapping(rule, written, members, spoken):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert parser.tokens[0].members == members
    assert rule.normalize(written) == spoken
    result = rule.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert result.mappings[0].input_text == written


@pytest.mark.parametrize("written", ["nvidia.com", "https://www.nvidia.com"])
def test_known_ending_has_one_verbalization(rule, written):
    tagged = rule.token_parser().reorder(rule.tag(written))
    lattice = (pynini.escape(tagged) @ rule.verbalizer).project("output").rmepsilon()
    outputs = set(pynini.shortestpath(lattice, nshortest=8, unique=True).paths().ostrings())
    assert len(outputs) == 1


@pytest.mark.parametrize("written, spoken", [
    ('文"件.pdf', '文"件 닷 피디에프'),
    (r"文\件.png", "文\\件 닷 피엔지"),
    ('https://a.com/a"\\01.pdf', 'HTTPS colon 슬래시 슬래시 a 닷컴/a"\\영일 닷 피디에프'),
])
def test_escaped_filename_and_url_round_trip(rule, written, spoken):
    tagged = rule.tag(written)
    parser = rule.token_parser()
    parser.parse(tagged)
    domain = written.split("://", 1)[-1]
    assert parser.tokens[0].members["domain"] == domain
    assert rule.verbalize(parser.reorder(tagged)) == spoken
    assert rule.normalize(written) == spoken
    assert rule.normalize_with_mapping(written).output_text == spoken
