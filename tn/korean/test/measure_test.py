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

from tn.korean.rules import measure as measure_module
from tn.korean.rules.cardinal import Cardinal
from tn.korean.rules.decimal import Decimal
from tn.korean.rules.fraction import Fraction
from tn.korean.rules.measure import Measure


@pytest.fixture(scope="module")
def measure():
    cardinal = Cardinal()
    return Measure(cardinal, Decimal(cardinal), Fraction(cardinal))


def parse(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert [token.name for token in parser.tokens] == ["measure"]
    return parser.tokens[0]


@pytest.mark.parametrize("written, order, fields", [
    ("12kg", ["integer", "units"], {"integer": "12", "units": "kg"}),
    ("-3km", ["negative", "integer", "units"], {"negative": "-", "integer": "3", "units": "km"}),
    ("60 km / h", ["integer", "units"], {"integer": "60", "units": "km / h"}),
    ("마이너스3.1km", ["negative", "integer_part", "fractional_part", "units"],
     {"negative": "마이너스", "integer_part": "3", "fractional_part": "1", "units": "km"}),
    ("10과1/2km/h", ["integer_part", "numerator", "denominator", "units"],
     {"integer_part": "10과", "numerator": "1", "denominator": "2", "units": "km/h"}),
    ("3분의1m", ["denominator", "numerator", "units"], {"denominator": "3", "numerator": "1", "units": "m"}),
    ("30°", ["integer", "units"], {"integer": "30", "units": "°"}),
    ("4/7이m", ["numerator", "denominator", "particle", "units"],
     {"numerator": "4", "denominator": "7", "particle": "이", "units": "m"}),
])
def test_tag_keeps_written_fields(measure, written, order, fields):
    token = parse(measure, written)
    assert token.order == order
    assert token.members == fields


def test_token_order_reads_denominator_first():
    order = measure_module.TOKEN_ORDER["measure"]
    assert order.index("negative") < order.index("integer_part") < order.index("denominator")
    assert order.index("denominator") < order.index("numerator") < order.index("units")
    assert order[-1] == "units"


@pytest.mark.parametrize("written, spoken", [
    ("1kg", "일 킬로그램"),
    ("-3km", "마이너스 삼 킬로미터"),
    ("60km/h", "육십 킬로미터 퍼 시간"),
    ("60 km / h", "육십 킬로미터 퍼 시간"),
    ("5/s", "오 퍼 초"),
    ("0.03m", "영점영삼 미터"),
    ("-3.1km", "마이너스 삼점일 킬로미터"),
    ("2과3/4L", "이과 사분의 삼 리터"),
    ("10과1/2km/h", "십과 이분의 일 킬로미터 퍼 시간"),
    ("-3/4km", "마이너스 사분의 삼 킬로미터"),
    ("3분의1m", "삼분의 일 미터"),
    ("1/√2m", "루트 이분의 일 미터"),
    ("100%", "백 퍼센트"),
    ("4/7이m", "칠분의 사가 미터"),
    ("1/2가m", "이분의 일이 미터"),
])
def test_normalize_matches_mapping(measure, written, spoken):
    assert measure.normalize(written) == spoken
    result = measure.normalize_with_mapping(written)
    assert result.output_text == spoken
    assert [(m.input_text, m.output_text) for m in result.mappings] == [(written, spoken)]


@pytest.mark.parametrize("written", ['1"kg', "1\\kg", '1kg"', "kg", "1", "1kg/"])
def test_rejects_quotes_backslashes_and_bare_parts(measure, written):
    with pytest.raises(RuntimeError):
        measure.normalize(written)
