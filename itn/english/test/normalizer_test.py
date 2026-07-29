# Copyright (c) 2026 Zhendong Peng (pzd17@tsinghua.org.cn)
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
from pynini import accep, escape

from itn.english.inverse_normalizer import InverseNormalizer
from itn.english.test.utils import parse_test_case
from tn.processor import _UniqueOutputPathStream
from tn.token_parser import TokenParser


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("en_itn")
    return InverseNormalizer(cache_dir=cache_dir, overwrite_cache=True)


class TestNormalizer:

    normalizer_cases = chain(
        parse_test_case("data/en_cardinal.txt"),
        parse_test_case("data/en_ordinal.txt"),
        parse_test_case("data/en_decimal.txt"),
        parse_test_case("data/en_date.txt"),
        parse_test_case("data/en_time.txt"),
        parse_test_case("data/en_money.txt"),
        parse_test_case("data/en_measure.txt"),
        parse_test_case("data/en_telephone.txt"),
        parse_test_case("data/en_electronic.txt"),
        parse_test_case("data/en_whitelist.txt"),
        parse_test_case("data/en_word.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written

    @pytest.mark.parametrize(
        "spoken,written,token_type,raw_fields",
        [
            ("twenty three", "23", "cardinal", {
                "integer": "twenty three"
            }),
            (
                "july twenty fifth two thousand twelve",
                "july 25 2012",
                "date",
                {
                    "month": "july",
                    "day": "twenty fifth",
                    "year": "two thousand twelve",
                },
            ),
            (
                "three point one four",
                "3.14",
                "decimal",
                {
                    "integer_part": "three",
                    "fractional_part": "point one four"
                },
            ),
            (
                "two dollars",
                "$2",
                "money",
                {
                    "value": "two",
                    "currency": "dollars"
                },
            ),
        ],
    )
    def test_raw_tags_and_exact_mapping(self, normalizer, spoken, written, token_type, raw_fields):
        parser = TokenParser("itn")
        parser.parse(normalizer.tag(spoken))
        assert len(parser.tokens) == 1
        token = parser.tokens[0]
        assert token.name == token_type
        for field, value in raw_fields.items():
            assert token.members[field] == value

        result = normalizer.normalize_with_mapping(spoken)
        assert result.output_text == written
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.token_type == token_type
        assert mapping.input_text == spoken
        assert mapping.output_text == written
        assert (mapping.input_start, mapping.input_end) == (0, len(spoken))
        assert (mapping.output_start, mapping.output_end) == (0, len(written))

    def test_nbest_and_mapping_share_joint_candidates(self, normalizer):
        outputs = normalizer.normalize("one hundred", nbest=3)
        results = normalizer.normalize_with_mapping("one hundred", nbest=3)
        assert [result.output_text for result in results] == outputs
        assert len(outputs) == len(set(outputs))

    def test_external_spaces_are_not_part_of_semantic_mapping(self, normalizer):
        result = normalizer.normalize_with_mapping("about twenty three dollars today", include_identity=True)
        money = next(mapping for mapping in result.mappings if mapping.token_type == "money")
        assert money.input_text == "twenty three dollars"
        assert money.output_text == "$23"

    def test_ip_priority_is_applied_once_outside_raw_field(self, normalizer):
        spoken = "one dot two dot three dot four"
        stream = _UniqueOutputPathStream(accep(escape(spoken)) @ normalizer.tagger)
        first = stream.pop()
        second = stream.pop()
        parser = TokenParser("itn")
        parser.parse(first.text)

        assert parser.tokens[0].name == "telephone"
        assert parser.tokens[0].members["kind"] == "ip"
        assert first.weight == pytest.approx(1.099, abs=1e-5)
        assert second.weight >= 1.1 - 1e-5
        assert normalizer.normalize(spoken) == "1.2.3.4"
