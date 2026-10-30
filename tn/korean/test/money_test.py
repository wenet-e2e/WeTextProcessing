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
from tn.korean.rules.money import TOKEN_ORDER, Money


@pytest.fixture(scope="module")
def money():
    return Money(Cardinal())


def parse(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert len(parser.tokens) == 1
    token = parser.tokens[0]
    assert token.name == "money"
    return token


@pytest.mark.parametrize("written, order, members", [
    ("₩2,000", ["currency_maj", "integer_part"], {"currency_maj": "₩", "integer_part": "2,000"}),
    ("KRW 1230000", ["currency_maj", "integer_part"], {"currency_maj": "KRW", "integer_part": "1230000"}),
    ("US$ 1,234,567", ["currency_maj", "integer_part"], {"currency_maj": "US$", "integer_part": "1,234,567"}),
    ("$ 30억", ["currency_maj", "integer_part"], {"currency_maj": "$", "integer_part": "30억"}),
    ("2000원", ["integer_part", "currency_maj"], {"integer_part": "2000", "currency_maj": "원"}),
    ("200달러", ["integer_part", "currency_maj"], {"integer_part": "200", "currency_maj": "달러"}),
])
def test_tag_keeps_written_fields(money, written, order, members):
    token = parse(money, written)
    assert token.order == order
    assert token.members == members
    for value in token.members.values():
        assert value in written


def test_prefix_currency_is_read_after_amount(money):
    assert TOKEN_ORDER == {"money": ["integer_part", "currency_maj", "minor_part"]}
    reordered = money.token_parser().reorder(money.tag("€500"))
    assert reordered == 'money { integer_part: "500" currency_maj: "€" }'
    assert money.normalize("€500") == "오백 유로"


@pytest.mark.parametrize("written, spoken", [
    ("₩2000", "이천원"),
    ("₩2,000", "이천원"),
    ("2000원", "이천원"),
    ("KRW 1230000", "백이십삼만원"),
    ("krw 2,345,600,000", "이십삼억사천오백육십만원"),
    ("₩0", "영원"),
    ("₩1,000억", "천억원"),
    ("₩12조", "십이조원"),
    ("₩1만", "일만원"),
    ("₩10.25", "십원"),
    ("$10.25", "십 달러"),
    ("$ 0", "영 달러"),
    ("US$ 1,234,567", "백이십삼만사천오백육십칠 달러"),
    ("$ 30억", "삼십억 달러"),
    ("HK$ 300", "삼백 홍콩 달러"),
    ("¥30조", "삼십조 엔"),
    ("AED 75", "칠십오 아랍에미리트 디르함"),
    ("200달러", "이백 달러"),
])
def test_normalize_matches_mapping(money, written, spoken):
    assert money.normalize(written) == spoken
    assert money.normalize_with_mapping(written).output_text == spoken


def test_minor_part_keeps_original_digits(money):
    token = parse(money, "₩10.25")
    assert token.order == ["currency_maj", "integer_part", "minor_part"]
    assert token.members == {"currency_maj": "₩", "integer_part": "10", "minor_part": "25"}


@pytest.mark.parametrize("written", ["₩1,00", "₩01", "₩", "₩10.2"])
def test_rejects_malformed_amounts(money, written):
    with pytest.raises(RuntimeError):
        money.normalize(written)
