# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import pytest

from itn.korean.inverse_normalizer import InverseNormalizer
from itn.korean.rules.cardinal import Cardinal
from itn.korean.rules.decimal import Decimal
from itn.korean.rules.fraction import Fraction
from itn.korean.rules.measure import Measure


DATA = Path(__file__).parent / "data"
CASES = [
    (path.stem, source, expected)
    for path in sorted(DATA.glob("test_cases_*.txt"))
    for line in path.open(encoding="utf-8")
    for source, expected in [line.rstrip("\n").split("~")]
]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


@pytest.mark.parametrize("category,source,expected", CASES)
def test_source_cases(normalizer, category, source, expected):
    assert normalizer.normalize(source) == expected


def test_source_case_count():
    assert len(CASES) == 209


@pytest.mark.parametrize("category,source,expected", CASES)
def test_source_mapping(normalizer, category, source, expected):
    result = normalizer.normalize_with_mapping(source, include_identity=True)
    assert result.output_text == expected
    for mapping in result.mappings:
        assert source[mapping.input_start:mapping.input_end] == mapping.input_text
        assert expected[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("source,expected", [
    ("네개를", "4개를"),
    ("여섯명만", "6명만"),
    ("열두개", "12개"),
    ("두명부터", "2명부터"),
    ("이분의일 만", "1/2 만"),
    ("이분의일만", "10000/2"),
    ("마이너스이와십분의일만", "-2 10000/10"),
    ("루트구분의이부터", "2/√9 부터"),
    ("마이너스십 킬로그램", "-10 kg"),
    ("마이너스 십점오 도", "-10.5 °"),
    ("일점공오 미터", "1.05 m"),
    ("루트구분의이 미터", "2/√9 m"),
    ("십 미터당초", "10 m/s"),
    ("일점오 달러", "$1.5"),
    ("마이너스일점오", "-1.5"),
    ("삼분", "03"),
    ("오초", "05"),
    ("삼분 오초", "03:05"),
    ("스물세시", "23:00"),
    ("더하기", "+"),
    ("왼쪽 괄호", "("),
    ("영영", "0 0"),
])
def test_additional_semantics(normalizer, source, expected):
    assert normalizer.normalize(source) == expected


@pytest.mark.parametrize("source,kind,members", [
    ("마이너스십칠", "cardinal", {"negative": "마이너스", "integer": "십칠"}),
    ("이천년유월십일", "date", {"year": "이천", "month": "유", "day": "십"}),
    ("일점오만", "decimal", {"integer_part": "일", "fractional_part": "오", "quantity": "만"}),
    ("마이너스이와십분의일", "fraction", {
        "negative": "마이너스", "integer_part": "이와", "denominator": "십", "numerator": "일",
    }),
    ("마이너스십 킬로그램", "measure", {"negative": "마이너스", "value": "십", "units": "킬로그램"}),
    ("일점오 달러", "money", {"value": "일점오", "currency": "달러"}),
    ("두개를", "ordinal", {"integer": "두", "counter": "개", "suffix": "를"}),
    ("공이에삼사오에육칠팔구", "telephone", {"number_part": "공이에삼사오에육칠팔구"}),
    ("오후 두시 반", "time", {"prefix": "오후", "hours": "두", "minutes": "반"}),
    ("더하기", "whitelist", {"value": "더하기"}),
    ("한국어", "word", {"value": "한국어"}),
])
def test_raw_fields(normalizer, source, kind, members):
    tagged = normalizer.tag(source)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == kind
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(source)


@pytest.mark.parametrize("size", [1, 32, 257])
def test_unbounded_fraction_swap(normalizer, size):
    denominator = "영점" + "일이삼" * size
    numerator = "이점" + "오육칠" * (size + 1)
    source = "마이너스" + denominator + "분의" + numerator
    expected = "-2." + "567" * (size + 1) + "/0." + "123" * size
    result = normalizer.normalize_with_mapping(source)
    assert result.output_text == expected
    assert len(result.mappings) == 1
    assert result.mappings[0].input_text == source
    assert result.mappings[0].token_type == "fraction"


@pytest.mark.parametrize("source", ["영영영", "박사", "이분의일", "일점오 달러", "한글\\경로\"{x}"])
def test_nbest_mapping_synchronization(normalizer, source):
    outputs = normalizer.normalize(source, nbest=3)
    mapped = normalizer.normalize_with_mapping(source, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(source)
    for result in mapped:
        for mapping in result.mappings:
            assert source[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_escaping_and_multibyte_offsets(normalizer):
    source = '한글\\경로"{x} 오천원'
    result = normalizer.normalize_with_mapping(source, include_identity=True)
    assert result.output_text == '한글\\경로"{x} ₩5000'
    assert [(m.input_text, m.output_text, m.token_type) for m in result.mappings] == [
        ('한글\\경로"{x}', '한글\\경로"{x}', "word"),
        ("오천원", "₩5000", "money"),
    ]
    assert result.mappings[1].input_start == source.index("오천원")
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(source))
    assert parser.tokens[0].members["value"] == '한글\\경로"{x}'


def test_zero_is_three_tokens(normalizer):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag("영영영"))
    assert [token.members for token in parser.tokens] == [{"integer": "영"}] * 3
    assert normalizer.normalize("영영영") == "0 0 0"


def test_mixed_fraction_has_only_regular_spaces(normalizer):
    result = normalizer.normalize("마이너스이와십분의일 만")
    assert result == "-2 1/10 만"
    assert "\u00a0" not in result


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("") == normalizer.normalize_with_mapping("").output_text
    assert normalizer.normalize("", nbest=3) == [""]


@pytest.mark.parametrize("source", [" ", "\t", "\n", " \t\n "])
def test_whitespace_only_input(normalizer, source):
    tagged = normalizer.tag(source)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": source}
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(source) == ""
    assert normalizer.normalize(source, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(source, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)


@pytest.mark.parametrize("kind,source,expected", [
    ("fraction", "마이너스이와십분의일", "-2 1/10"),
    ("measure", "마이너스십 킬로그램", "-10 kg"),
])
def test_standalone_reordered_rule(kind, source, expected):
    cardinal = Cardinal()
    rule = Fraction(cardinal, Decimal(cardinal)) if kind == "fraction" else Measure(cardinal)
    assert rule.normalize(source) == expected
    assert rule.normalize_with_mapping(source).output_text == expected
