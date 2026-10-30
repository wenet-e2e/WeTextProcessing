# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from tn.korean.rules import fraction as fraction_module
from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.fraction import Fraction


@pytest.fixture(scope="module")
def fraction():
    return Fraction(Cardinal())


def parse(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert [token.name for token in parser.tokens] == ["fraction"]
    return parser.tokens[0]


@pytest.mark.parametrize("written, order, fields", [
    ("1/2", ["numerator", "denominator"], {"numerator": "1", "denominator": "2"}),
    ("3분의1", ["denominator", "numerator"], {"denominator": "3", "numerator": "1"}),
    ("-1과√1/2", ["negative", "integer_part", "numerator", "denominator"],
     {"negative": "-", "integer_part": "1과", "numerator": "√1", "denominator": "2"}),
    ("마이너스1과1/√3", ["negative", "integer_part", "numerator", "denominator"],
     {"negative": "마이너스", "integer_part": "1과", "numerator": "1", "denominator": "√3"}),
    ("4/7이", ["numerator", "denominator", "particle"], {"numerator": "4", "denominator": "7", "particle": "이"}),
    ("1.5/2를", ["numerator", "denominator", "particle"], {"numerator": "1.5", "denominator": "2", "particle": "를"}),
])
def test_tag_keeps_written_fields(fraction, written, order, fields):
    token = parse(fraction, written)
    assert token.order == order
    assert token.members == fields


def test_token_order_reads_denominator_first():
    assert fraction_module.TOKEN_ORDER == {
        "fraction": ["negative", "integer_part", "denominator", "numerator", "particle"]
    }


@pytest.mark.parametrize("written, spoken", [
    ("1/2", "이분의 일"),
    ("121분의3221", "백이십일분의 삼천이백이십일"),
    ("2와12/33", "이와 삼십삼분의 십이"),
    ("-1과√1/2", "마이너스 일과 이분의 루트 일"),
    ("1과1/√3", "일과 루트 삼분의 일"),
    ("√2와1/2", "루트 이와 이분의 일"),
    ("1/2.5", "이점오분의 일"),
    ("3/5이", "오분의 삼이"),
    ("4/7이", "칠분의 사가"),
    ("2/3은", "삼분의 이는"),
    ("1/3를", "삼분의 일을"),
    ("5/9을", "구분의 오를"),
    ("1/0.22이", "영점이이분의 일이"),
    ("0.22/3", "삼분의 영점이이"),
    ("0.22/3이", "삼분의 영점이이가"),
])
def test_normalize_matches_mapping(fraction, written, spoken):
    assert fraction.normalize(written) == spoken
    result = fraction.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert [(m.input_text, m.output_text) for m in result.mappings] == [(written, spoken)]


@pytest.mark.parametrize("written", ['1/"2', "1/\\2", '"1/2', "1/", "/2"])
def test_rejects_quotes_backslashes_and_partial_fractions(fraction, written):
    with pytest.raises(RuntimeError):
        fraction.normalize(written)


def test_exposes_bare_semantic_graphs(fraction):
    assert fraction.sign is not None
    for graph in (fraction.numeral, fraction.integer_graph):
        assert graph.num_states() > 0
