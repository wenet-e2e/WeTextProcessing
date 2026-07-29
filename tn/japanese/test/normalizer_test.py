# Copyright (c) 2024 Logan Liu (2319277867@qq.com)
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

from itertools import chain

import pytest

from tn.japanese.normalizer import Normalizer
from tn.japanese.test.utils import parse_test_case


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("ja_tn")
    return Normalizer(cache_dir=cache_dir, overwrite_cache=True)


@pytest.fixture(scope="module")
def normalizer_without_full_to_half(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("ja_tn_no_full_to_half")
    return Normalizer(
        cache_dir=cache_dir,
        overwrite_cache=True,
        full_to_half=False,
    )


class TestNormalizer:

    normalizer_cases = chain(
        parse_test_case("data/cardinal.txt"),
        parse_test_case("data/char.txt"),
        parse_test_case("data/date.txt"),
        parse_test_case("data/fraction.txt"),
        parse_test_case("data/math.txt"),
        parse_test_case("data/measure.txt"),
        parse_test_case("data/money.txt"),
        parse_test_case("data/sport.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written


class TestRawFieldContract:

    @pytest.mark.parametrize(
        "source, tagged",
        [
            ("12", 'cardinal { value: "12" }'),
            ("1/3", 'fraction { numerator: "1" denominator: "3" }'),
            ("1-2=-1", 'math { value: "1-2=-1" }'),
            ("10km/h", 'measure { value: "10km/h" }'),
            ("USD1001", 'money { currency: "USD" value: "1001" }'),
            ("2:3", 'sport { score: "2:3" }'),
            ("3:02", 'time { hour: "3" minute: "02" }'),
            ("B2B", 'whitelist { value: "B2B" }'),
        ],
    )
    def test_semantic_tags_preserve_written_values(self, normalizer, source, tagged):
        assert normalizer.tag(source) == tagged

    def test_date_tag_preserves_fullwidth_fields(self, normalizer):
        assert normalizer.tag("２０２２／１２／１") == 'date { year: "２０２２" month: "１２" day: "１" }'

    def test_time_range_tag_preserves_separator(self, normalizer):
        assert normalizer.tag("3:30-4:34") == ('time { hour: "3" minute: "30" } '
                                               'range { value: "-" } '
                                               'time { hour: "4" minute: "34" }')

    def test_mapping_uses_raw_measure_source(self, normalizer):
        result = normalizer.normalize_with_mapping("今日は12時です")

        assert result.output_text == "今日は十二時です"
        assert [mapping.as_dict() for mapping in result.mappings] == [{
            "kind": "replace",
            "token_type": "measure",
            "input": {
                "start": 3,
                "end": 6,
                "text": "12時",
            },
            "output": {
                "start": 3,
                "end": 6,
                "text": "十二時",
            },
        }]

    def test_mapping_tracks_fullwidth_characters_and_date(self, normalizer):
        result = normalizer.normalize_with_mapping("Ａ，２０２２／１２／１")

        assert result.output_text == "A,二千二十二年十二月一日"
        assert [(mapping.token_type, mapping.input_text, mapping.output_text) for mapping in result.mappings] == [
            ("char", "Ａ", "A"),
            ("char", "，", ","),
            ("date", "２０２２／１２／１", "二千二十二年十二月一日"),
        ]

    def test_fullwidth_colon_remains_a_time_without_canonicalization(self, normalizer_without_full_to_half):
        assert normalizer_without_full_to_half.tag("3：30") == 'time { hour: "3" minute: "30" }'
        assert normalizer_without_full_to_half.normalize("3：30") == "三時三十分"

    @pytest.mark.parametrize(
        "source, token_type, token_source",
        [
            ("3:02 次", "time", "3:02"),
            ("2:3 次", "sport", "2:3"),
        ],
    )
    def test_semantic_tokens_do_not_consume_external_spaces(self, normalizer, source, token_type, token_source):
        result = normalizer.normalize_with_mapping(source)
        mapping = next(mapping for mapping in result.mappings if mapping.token_type == token_type)

        assert mapping.input_text == token_source

    def test_time_wins_after_fullwidth_date_and_external_space(self, normalizer):
        source = "２０２２／１２／１ 3:30"

        assert normalizer.tag(source).endswith('time { hour: "3" minute: "30" }')
        assert normalizer.normalize(source) == "二千二十二年十二月一日 三時三十分"

    def test_raw_score_excludes_surrounding_spaces(self, normalizer):
        tagged = normalizer.tag("abc 2:3 def")

        assert 'sport { score: "2:3" }' in tagged
        assert 'score: " 2:3"' not in tagged
        assert 'score: "2:3 "' not in tagged

    def test_fullwidth_fraction_separator_respects_canonicalization_flag(self, normalizer, normalizer_without_full_to_half):
        source = "1／3"

        assert normalizer.tag(source) == 'fraction { numerator: "1" denominator: "3" }'
        assert normalizer.normalize(source) == "三分の一"
        assert normalizer_without_full_to_half.tag(source) == (
            'cardinal { value: "1" } char { value: "／" } cardinal { value: "3" }')
        assert normalizer_without_full_to_half.normalize(source) == "一／三"
