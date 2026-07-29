# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

from tn.chinese.normalizer import Normalizer
from tn.chinese.test.utils import parse_test_case


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("zh_tn")
    return Normalizer(cache_dir=cache_dir, overwrite_cache=True, tag_oov=True)


class TestNormalizer:

    normalizer_cases = chain(
        parse_test_case("data/cardinal.txt"),
        parse_test_case("data/char.txt"),
        parse_test_case("data/date.txt"),
        parse_test_case("data/fraction.txt"),
        parse_test_case("data/math.txt"),
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/normalizer.txt"),
        parse_test_case("data/normalizer_tag_oov.txt"),
    )

    @pytest.mark.parametrize("written, spoken", normalizer_cases)
    def test_normalizer(self, normalizer, written, spoken):
        assert normalizer.normalize(written) == spoken

    def test_normalize_with_mapping(self, normalizer):
        result = normalizer.normalize_with_mapping("今天中午12点")

        assert result.output_text == "今天中午十二点"
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.token_type == "math"
        assert (mapping.input_start, mapping.input_end, mapping.input_text) == (4, 6, "12")
        assert (mapping.output_start, mapping.output_end, mapping.output_text) == (4, 6, "十二")

    @pytest.mark.parametrize(
        "written,tagged",
        [
            ("今天中午12点", 'math { value: "12" }'),
            ("2024/01/02", 'date { year: "2024" month: "01" day: "02" }'),
            ("3/4", 'fraction { numerator: "3" denominator: "4" }'),
            ("￥12", 'money { currency: "￥" value: "12" }'),
            ("10km/h", 'measure { numerator: "10km" denominator: "h" }'),
            ("蘋果", 'char { value: "蘋" } char { value: "果" }'),
        ],
    )
    def test_tagger_preserves_written_fields(self, normalizer, written, tagged):
        assert tagged in normalizer.tag(written)

    def test_range_is_tagged_before_verbalization(self, normalizer):
        tagged = normalizer.tag("10:30-11:20")

        assert 'range { value: "-" }' in tagged
        result = normalizer.normalize_with_mapping("10:30-11:20")
        assert [mapping.token_type for mapping in result.mappings] == ["time", "range", "time"]
        assert result.mappings[1].input_text == "-"
        assert result.mappings[1].output_text == "到"

    def test_traditional_to_simple_happens_after_tagging(self, normalizer):
        result = normalizer.normalize_with_mapping("蘋果")

        assert result.output_text == "苹果"
        assert result.mappings[0].token_type == "char"
        assert result.mappings[0].input_text == "蘋"
        assert result.mappings[0].output_text == "苹"

    def test_single_unicode_token_mapping_uses_character_spans(self, normalizer):
        result = normalizer.normalize_with_mapping("蘋")

        assert result.output_text == "苹"
        assert len(result.mappings) == 1
        assert (result.mappings[0].input_start, result.mappings[0].input_end) == (0, 1)
        assert (result.mappings[0].output_start, result.mappings[0].output_end) == (0, 1)
