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
from pynini.lib import byte

from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.ordinal import Ordinal


@pytest.fixture(scope="module")
def ordinal():
    return Ordinal(Cardinal())


def members(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert [token.name for token in parser.tokens] == ["ordinal"]
    return parser.tokens[0].members


@pytest.mark.parametrize("written, fields", [
    ("1번째", {"integer": "1", "suffix": "번째"}),
    ("39번째", {"integer": "39", "suffix": "번째"}),
    ("1000번째부터", {"integer": "1000", "suffix": "번째", "particle": "부터"}),
    ("2번째로까지", {"integer": "2", "suffix": "번째", "particle": "로까지"}),
])
def test_tag_keeps_written_fields(ordinal, written, fields):
    assert members(ordinal, written) == fields


@pytest.mark.parametrize("written, spoken", [
    ("1번째", "첫번째"),
    ("11번째", "열한번째"),
    ("20번째", "스무번째"),
    ("29번째", "스물아홉번째"),
    ("30번째", "서른번째"),
    ("40번째", "사십번째"),
    ("189번째", "백팔십구번째"),
    ("1000000000번째", "십억번째"),
    ("3번째가", "세번째가"),
    ("21번째까지", "스물한번째까지"),
    ("5번째는부터", "다섯번째는부터"),
])
def test_normalize_matches_mapping(ordinal, written, spoken):
    assert ordinal.normalize(written) == spoken
    result = ordinal.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert [(m.input_text, m.output_text) for m in result.mappings] == [(written, spoken)]


@pytest.mark.parametrize("written", ['1"번째', "1\\번째", "번째", "1번", "04번째"])
def test_rejects_quotes_backslashes_and_bad_numbers(ordinal, written):
    with pytest.raises(RuntimeError):
        ordinal.normalize(written)


def test_native_range_uses_independent_single_digit_graph(monkeypatch):
    cardinal = Cardinal()
    # Other rules may call the mutating Fst.closure() on Processor.DIGIT.
    # Reproduce that broadened input alphabet without leaking shared state.
    monkeypatch.setattr(byte, "DIGIT", byte.DIGIT.copy().closure(0, 3))
    rule = Ordinal(cardinal)
    assert rule.normalize("1번째") == "첫번째"
    assert rule.normalize("21번째") == "스물한번째"
    assert rule.normalize("40번째") == "사십번째"
