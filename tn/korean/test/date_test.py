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
from tn.korean.rules.date import TOKEN_ORDER, Date


@pytest.fixture(scope="module")
def date():
    return Date(Cardinal())


def parse(rule, written):
    parser = rule.token_parser()
    parser.parse(rule.tag(written))
    assert len(parser.tokens) == 1
    token = parser.tokens[0]
    assert token.name == "date"
    return token


@pytest.mark.parametrize("written, order, members", [
    ("2024/01/30", ["year", "month", "day"], {"year": "2024", "month": "01", "day": "30"}),
    ("2024.01.30.(화)", ["year", "month", "day", "weekday"],
     {"year": "2024", "month": "01", "day": "30", "weekday": "화"}),
    ("10/30/2024", ["month", "day", "year"], {"month": "10", "day": "30", "year": "2024"}),
    ("29/12/2000", ["day", "month", "year"], {"day": "29", "month": "12", "year": "2000"}),
    ("2024년", ["year"], {"year": "2024년"}),
    ("기원전128년", ["era", "year"], {"era": "기원전", "year": "128년"}),
    ("1900년대", ["year"], {"year": "1900년대"}),
    ("28일수요일", ["day", "weekday"], {"day": "28일", "weekday": "수요일"}),
    ("1월1일(월)", ["month", "day", "weekday"], {"month": "1월", "day": "1일", "weekday": "월"}),
    ("9일(월〜금)", ["day", "weekday"], {"day": "9일", "weekday": "월〜금"}),
    ("7월5", ["month", "range_start"], {"month": "7월", "range_start": "5"}),
    ("3월에", ["month"], {"month": "3월에"}),
])
def test_tag_keeps_written_fields(date, written, order, members):
    token = parse(date, written)
    assert token.order == order
    assert token.members == members
    for value in token.members.values():
        assert value in written


@pytest.mark.parametrize("written, spoken", [
    ("10/30/2024", "이천이십사년 시월 삼십일"),
    ("29/12/2000", "이천년 십이월 이십구일"),
])
def test_day_first_and_month_first_formats_are_reordered(date, written, spoken):
    tagged = date.tag(written)
    assert date.token_parser().reorder(tagged).startswith('date { year: "')
    assert date.normalize(written) == spoken


def test_token_order_is_reading_order():
    assert TOKEN_ORDER["date"][:3] == ["era", "year", "month"]
    assert TOKEN_ORDER["date"].index("day") < TOKEN_ORDER["date"].index("weekday")


@pytest.mark.parametrize("written, spoken", [
    ("2024년", "이천이십사년 "),
    ("3월", "삼월 "),
    ("15일", "십오일"),
    ("2024/01/30", "이천이십사년 일월 삼십일"),
    ("2024-07-23", "이천이십사년 칠월 이십삼일"),
    ("1999.9.17", "천구백구십구년 구월 십칠일"),
    ("2024.01.30.(화)", "이천이십사년 일월 삼십일 화요일"),
    ("기원후1390년", "기원후 천삼백구십년 "),
    ("28일수요일", "이십팔일 수요일"),
    ("1900년대", "천구백년대 "),
    ("6월1일", "유월 일일"),
    ("10월8일(화)", "시월 팔일 화요일"),
    ("9일(월〜금)", "구일 월요일부터금요일"),
    ("7월5", "칠월 오"),
    ("2024년 3월 1일", "이천이십사년 삼월 일일"),
    ("(수)", "수요일"),
])
def test_normalize_matches_mapping(date, written, spoken):
    assert date.normalize(written) == spoken
    assert date.normalize_with_mapping(written).output_text == spoken


@pytest.mark.parametrize("written", ["2024/13/01", "2024/01/32", "13월", "32일", "2024/01-30", "0년"])
def test_rejects_invalid_dates(date, written):
    with pytest.raises(Exception):
        date.normalize(written)
