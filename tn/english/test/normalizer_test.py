# Copyright (c) 2024 Xingchen Song (sxc19@tsinghua.org.cn)
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

from tn.english.normalizer import Normalizer
from tn.english.test.utils import parse_test_case
from tn.token_parser import TokenParser


@pytest.fixture(scope="module")
def normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("en_tn")
    return Normalizer(cache_dir=cache_dir, overwrite_cache=True)


class TestNormalizer:

    cases = parse_test_case("data/normalizer.txt")

    @pytest.mark.parametrize("written, spoken", cases)
    def test_normalizer(self, normalizer, written, spoken):
        assert normalizer.normalize(written) == spoken

    def test_normalize_with_mapping_uses_full_stream_spacing(self, normalizer):
        result = normalizer.normalize_with_mapping("I have 23 apples", include_identity=True)

        assert result.output_text == "I have twenty three apples"
        assert [mapping.input_text for mapping in result.mappings] == ["I", "have", "23", "apples"]
        assert [mapping.output_text for mapping in result.mappings] == ["I", "have", "twenty three", "apples"]
        cardinal = result.mappings[2]
        assert cardinal.token_type == "cardinal"
        assert (cardinal.input_start, cardinal.input_end) == (7, 9)
        assert (cardinal.output_start, cardinal.output_end) == (7, 19)

    def test_normalize_with_mapping_preserves_punctuation_adjacency(self, normalizer):
        result = normalizer.normalize_with_mapping("Hello, world!", include_identity=True)

        assert result.output_text == "Hello, world!"
        assert [mapping.input_text for mapping in result.mappings] == ["Hello", ", ", "world", "!"]
        assert [mapping.output_text for mapping in result.mappings] == ["Hello", ", ", "world", "!"]
        assert [(mapping.output_start, mapping.output_end) for mapping in result.mappings] == [
            (0, 5),
            (5, 7),
            (7, 12),
            (12, 13),
        ]

    def test_cardinal_default_is_independent_of_sentence_context(self, normalizer):
        # The old spoken-field tagger exposed a whole-sentence shortest-path
        # tie: the same 256 selected different verbalizations in two fixtures.
        # Raw fields leave the locally weighted with-and default to verbalization.
        assert normalizer.normalize("number 256") == "number two hundred and fifty six"
        assert normalizer.normalize("before number 256 after") == "before number two hundred and fifty six after"

    @pytest.mark.parametrize(
        "written, token_type, expected_fields",
        [
            ("23", "cardinal", {
                "integer": "23"
            }),
            ("2024-05-06", "date", {
                "year": "2024",
                "month": "05",
                "day": "06"
            }),
            ("2:00", "time", {
                "hours": "2",
                "minutes": "00",
            }),
            ("03:43 p.m.", "time", {
                "hours": "03",
                "minutes": "43",
                "suffix": "p.m."
            }),
            ("$12.05", "money", {
                "currency_maj": "$",
                "integer_part": "12",
                "fractional_part": "05"
            }),
            ("$1", "money", {
                "currency_maj": "$",
                "integer_part": "1",
            }),
            ("$1.00", "money", {
                "currency_maj": "$",
                "integer_part": "1",
                "fractional_part": "00",
            }),
            ("$1.2000", "money", {
                "currency_maj": "$",
                "integer_part": "1",
                "fractional_part": "2000",
            }),
            ("31.990 billion", "decimal", {
                "integer_part": "31",
                "fractional_part": "990",
                "quantity": "billion",
            }),
            ("3/4", "fraction", {
                "numerator": "3",
                "denominator": "4"
            }),
            ("¾", "fraction", {
                "value": "¾"
            }),
            ("cdf1@abc.edu", "electronic", {
                "username": "cdf1",
                "domain": "abc.edu"
            }),
            ("123-123-5678-1", "telephone", {
                "number_part": "123-123-5678-",
                "extension": "1"
            }),
            ("A-123", "serial", {
                "name": "A-123"
            }),
        ],
    )
    def test_semantic_tags_preserve_written_fields(self, normalizer, written, token_type, expected_fields):
        parser = TokenParser("en_tn")
        parser.parse(normalizer.tag(written))

        assert len(parser.tokens) == 1
        token = parser.tokens[0]
        assert token.name == token_type
        for key, value in expected_fields.items():
            assert token.members[key] == value

    @pytest.mark.parametrize(
        "written, spoken, token_type",
        [
            ("23", "twenty three", "cardinal"),
            ("2024-05-06", "the sixth of may twenty twenty four", "date"),
            ("03:43 p.m.", "three forty three PM", "time"),
            ("$12.05", "twelve point oh five dollars", "money"),
            ("$1.00", "one dollar", "money"),
            ("$1.2000", "one point two dollars", "money"),
            ("31.990 billion", "thirty one point nine nine oh billion", "decimal"),
            ("3/4", "three quarters", "fraction"),
            ("1,000", "thousand", "cardinal"),
            ("¾", "three quarters", "fraction"),
        ],
    )
    def test_mapping_covers_semantic_rules_and_unicode(self, normalizer, written, spoken, token_type):
        result = normalizer.normalize_with_mapping(written, include_identity=True)

        assert result.output_text == spoken
        assert len(result.mappings) == 1
        mapping = result.mappings[0]
        assert mapping.token_type == token_type
        assert mapping.input_text == written
        assert mapping.output_text == spoken
        assert (mapping.input_start, mapping.input_end) == (0, len(written))
        assert (mapping.output_start, mapping.output_end) == (0, len(spoken))
