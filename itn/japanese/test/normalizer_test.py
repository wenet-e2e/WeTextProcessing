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

from itn.japanese.inverse_normalizer import InverseNormalizer
from itn.japanese.test.utils import parse_test_case
from tn.token_parser import TokenParser


@pytest.fixture(scope="class")
def normalizer(request, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("ja_itn")
    return InverseNormalizer(cache_dir=cache_dir, overwrite_cache=True, **request.param)


@pytest.fixture(scope="module")
def fullwidth_normalizer(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("ja_itn_fullwidth")
    return InverseNormalizer(
        cache_dir=cache_dir,
        overwrite_cache=True,
        full_to_half=True,
    )


@pytest.mark.parametrize(
    "normalizer",
    [{
        "enable_standalone_number": True,
        "enable_0_to_9": True,
        "enable_million": False,
    }],
    indirect=True,
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
        parse_test_case("data/number.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written

    @pytest.mark.parametrize(
        "spoken,written,token_type,raw_fields",
        [
            ("二十三", "23", "cardinal", {
                "value": "二十三"
            }),
            (
                "二千二十四年十月一日",
                "2024年10月1日",
                "date",
                {
                    "year": "二千二十四年",
                    "month": "十月",
                    "day": "一日"
                },
            ),
            ("四分の三", "3/4", "fraction", {
                "denominator": "四分の",
                "numerator": "三"
            }),
            (
                "三千三百八十点五八ドル",
                "$3380.58",
                "money",
                {
                    "value": "三千三百八十点五八",
                    "currency": "ドル"
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


@pytest.mark.parametrize(
    "normalizer",
    [{
        "enable_standalone_number": False,
        "enable_0_to_9": True,
        "enable_million": False,
    }],
    indirect=True,
)
class TestNormalizerDisablestandalonenumberEnable0to9:

    normalizer_cases = chain(
        parse_test_case("data/char.txt"),
        parse_test_case("data/date.txt"),
        parse_test_case("data/fraction.txt"),
        parse_test_case("data/math.txt"),
        parse_test_case("data/measure.txt"),
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/normalizer_disable_standalone_number_enable_0_to_9.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize(
    "normalizer",
    [{
        "enable_standalone_number": True,
        "enable_0_to_9": False,
        "enable_million": False,
    }],
    indirect=True,
)
class TestNormalizerEnablestandalonenumberDisable0to9:

    normalizer_cases = chain(
        parse_test_case("data/char.txt"),
        parse_test_case("data/date.txt"),
        parse_test_case("data/fraction.txt"),
        parse_test_case("data/math.txt"),
        parse_test_case("data/measure.txt"),
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/normalizer_enable_standalone_number_disable_0_to_9.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize(
    "normalizer",
    [{
        "enable_standalone_number": False,
        "enable_0_to_9": False,
        "enable_million": False,
    }],
    indirect=True,
)
class TestNormalizerDisablestandalonenumberDisable0to9:

    normalizer_cases = chain(
        parse_test_case("data/char.txt"),
        parse_test_case("data/date.txt"),
        parse_test_case("data/fraction.txt"),
        parse_test_case("data/math.txt"),
        parse_test_case("data/measure.txt"),
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/normalizer_disable_standalone_number_disable_0_to_9.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize(
    "source,written",
    [
        ('"', '"'),
        ("\\", "\\"),
        ("＂", '"'),
        ("＼", "\\"),
    ],
)
def test_quoted_field_escaping_preserves_raw_mapping(fullwidth_normalizer, source, written):
    parser = TokenParser("itn")
    parser.parse(fullwidth_normalizer.tag(source))
    assert parser.tokens[0].members["value"] == source

    result = fullwidth_normalizer.normalize_with_mapping(source, include_identity=True)
    assert result.output_text == written
    assert len(result.mappings) == 1
    assert result.mappings[0].input_text == source
    assert result.mappings[0].output_text == written
