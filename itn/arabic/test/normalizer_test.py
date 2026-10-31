# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import pytest

from itn.arabic.inverse_normalizer import InverseNormalizer


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        # Universal newlines, as in the original tests.
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                spoken, written = line.split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def parse(normalizer, spoken):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    return [(token.name, token.members) for token in parser.tokens]


def test_original_fixture_count():
    assert len(CASES) == 126
    assert CASES[0].values == ("ألفان وتسعة", "2009")


@pytest.mark.parametrize("spoken, written", CASES)
def test_original(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", CASES)
def test_original_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        spoken, written = param.values
        assert normalizer.verbalize(normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken, expected", [
    ("تسعة اَلاف وثمان مئة وسبعة وثلاثون", [("cardinal", {"integer": "تسعة اَلاف وثمان مئة وسبعة وثلاثون"})]),
    ("سالب خمسة", [("cardinal", {"negative": "سالب", "integer": "خمسة"})]),
    ("سبعة وتسعة وتسعون من مئة", [("decimal", {"integer_part": "سبعة", "fractional_part": "تسعة وتسعون من مئة"})]),
    ("واحد و نصف", [("fraction", {"integer_part": "واحد", "denominator": "نصف"})]),
    ("ستة عشر و سبعة عشر على ثلاثة", [("fraction", {
        "integer_part": "ستة عشر",
        "numerator": "سبعة عشر",
        "denominator": "ثلاثة"
    })]),
    ("سبعة دولار وخمسون سنت", [("money", {
        "integer_part": "سبعة",
        "currency": "دولار",
        "fractional_part": "خمسون",
        "currency_min": "سنت"
    })]),
    ("عشرون في المائة", [("measure", {"integer": "عشرون", "units": "في المائة"})]),
])
def test_raw_field_tags(normalizer, spoken, expected):
    assert parse(normalizer, spoken) == expected


@pytest.mark.parametrize("spoken, written", [
    ("سالب خمسة", "-5"),
    ("ثلاثة اَلاف ومئة", "3100"),
    ("لدي ثلاثة كتب", "لدي 3 كتب"),
    ("خمسة ريال سعودي", "ر.س5"),
    ("سالب ستة وثمانية من عشرة", "-6.8"),
])
def test_hand_written(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["ثلاثة اَلاف وستة", "ستة وثمانية وثمانون من مئة", "واحد و نصف", "سبعة دولار وخمسون سنت"])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)


def test_escaped_word_round_trips(normalizer):
    spoken = 'مسار\\a"b'
    assert parse(normalizer, spoken) == [("word", {"value": spoken})]
    assert normalizer.normalize(spoken) == spoken


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, spoken):
    tagged = normalizer.tag(spoken)
    assert parse(normalizer, spoken) == [("word", {"value": spoken})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
