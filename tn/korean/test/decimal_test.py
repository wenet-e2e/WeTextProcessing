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

from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.decimal import Decimal


@pytest.fixture(scope="module")
def decimal():
    return Decimal(Cardinal())


def members(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert [token.name for token in parser.tokens] == ["decimal"]
    return parser.tokens[0].members


@pytest.mark.parametrize("written, fields", [
    ("0.5", {"integer_part": "0", "fractional_part": "5"}),
    ("-11.99", {"negative": "-", "integer_part": "11", "fractional_part": "99"}),
    ("마이너스2.5", {"negative": "마이너스", "integer_part": "2", "fractional_part": "5"}),
    ("1000.01", {"integer_part": "1000", "fractional_part": "01"}),
])
def test_tag_keeps_written_fields(decimal, written, fields):
    assert members(decimal, written) == fields


@pytest.mark.parametrize("written, spoken", [
    ("0.1", "영점일"),
    ("-0.5", "마이너스 영점오"),
    ("마이너스2.5", "마이너스 이점오"),
    ("1000.01", "천점영일"),
    ("10000.5", "만점오"),
    ("123456.2234", "십이만삼천사백오십육점이이삼사"),
])
def test_normalize_matches_mapping(decimal, written, spoken):
    assert decimal.normalize(written) == spoken
    result = decimal.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert [(m.input_text, m.output_text) for m in result.mappings] == [(written, spoken)]


@pytest.mark.parametrize("written", ['1."5', "1.\\5", '"1.5', "1.", ".5"])
def test_rejects_quotes_backslashes_and_partial_numbers(decimal, written):
    with pytest.raises(RuntimeError):
        decimal.normalize(written)


def test_exposes_bare_semantic_graphs(decimal):
    assert decimal.sign is not None
    for graph in (decimal.integer_graph, decimal.fractional_graph, decimal.graph):
        assert graph.num_states() > 0
