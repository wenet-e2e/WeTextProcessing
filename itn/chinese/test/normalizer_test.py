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
from pynini import FstOpError, accep, escape

from itn.chinese.inverse_normalizer import InverseNormalizer
from itn.chinese.rules.cardinal import Cardinal
from itn.chinese.rules.time import Time
from itn.chinese.test.utils import parse_test_case
from tn.processor import _UniqueOutputPathStream
from tn.token_parser import TokenParser


@pytest.fixture(scope="class")
def normalizer(request, tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("zh_itn")
    return InverseNormalizer(cache_dir=cache_dir, overwrite_cache=True, **request.param)


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
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/number.txt"),
        parse_test_case("data/license_plate.txt"),
        parse_test_case("data/train_number.txt"),
        parse_test_case("data/normalizer.txt"),
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
                "二零零八年八月八日",
                "2008/08/08",
                "date",
                {
                    "year": "二零零八年",
                    "month": "八月",
                    "day": "八日"
                },
            ),
            ("三点一四一五", "3.1415", "cardinal", {
                "value": "三点一四一五"
            }),
            (
                "三千三百八十元五角八分",
                "¥3380.58",
                "money",
                {
                    "value": "三千三百八十",
                    "currency": "元",
                    "decimal": "五角八分"
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
        assert (mapping.input_start, mapping.input_end, mapping.input_text) == (
            0,
            len(spoken),
            spoken,
        )
        assert (mapping.output_start, mapping.output_end, mapping.output_text) == (
            0,
            len(written),
            written,
        )

    @pytest.mark.parametrize(
        "spoken,written,token_type,raw_value",
        [
            ("一度", "1°", "measure", "一度"),
            ("百分之三十", "30%", "measure", "百分之三十"),
        ],
    )
    def test_joint_candidates_keep_raw_field_and_semantic_weight(self, normalizer, spoken, written, token_type, raw_value):
        candidate = normalizer._normalization_candidates(spoken, nbest=1)[0]
        parser = TokenParser("itn")
        parser.parse(candidate.tagged)

        assert candidate.output == written
        assert parser.tokens[0].name == token_type
        assert parser.tokens[0].members["value"] == raw_value

    def test_bare_zero_minute_requires_noon_prefix(self, normalizer):
        with pytest.raises(FstOpError):
            Time().tag("一点零二")
        assert normalizer.normalize("一点零二") == "1.02"
        assert normalizer.normalize("早上一点零二") == "1:02a.m."

    def test_cardinal_raw_field_keeps_minimum_input_weight(self, normalizer):
        cardinal = Cardinal(
            enable_standalone_number=True,
            enable_0_to_9=True,
            enable_million=False,
        )
        stream = _UniqueOutputPathStream(accep(escape("一二")) @ cardinal.tagger)
        tagged = stream.pop()
        parser = TokenParser("itn")
        parser.parse(tagged.text)
        candidates = cardinal._normalization_candidates("一二", nbest=2)

        assert tagged.weight == pytest.approx(0.1)
        assert parser.tokens[0].members["value"] == "一二"
        assert [candidate.tagger_weight for candidate in candidates] == pytest.approx([0.1, 0.1])
        assert [candidate.weight for candidate in candidates] == pytest.approx([0.1, 0.3])


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
        parse_test_case("data/license_plate.txt"),
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
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/license_plate.txt"),
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
        parse_test_case("data/money.txt"),
        parse_test_case("data/time.txt"),
        parse_test_case("data/whitelist.txt"),
        parse_test_case("data/license_plate.txt"),
        parse_test_case("data/normalizer_disable_standalone_number_disable_0_to_9.txt"),
    )

    @pytest.mark.parametrize("spoken, written", normalizer_cases)
    def test_normalizer(self, normalizer, spoken, written):
        assert normalizer.normalize(spoken) == written
