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

from tn.arabic.normalizer import Normalizer

# The original fraction and measure TN tests read their fixtures as
# ``spoken~written``; the other files are ``written~spoken``. The whitelist
# file is shared with ITN and is used unchanged in both directions.
SPOKEN_FIRST = {"test_cases_fraction", "test_cases_measure"}


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        # Universal newlines, as in the original tests: CRLF fixtures parse as LF.
        with path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                first, second = line.split("~")
                written, spoken = (second, first) if path.stem in SPOKEN_FIRST else (first, second)
                cases.append(pytest.param(written, spoken, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def parse(normalizer, written):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    return [(token.name, token.members) for token in parser.tokens]


def test_original_fixture_count():
    assert len(CASES) == 124
    assert CASES[27].values == ("6.88", "ستة وثمانية وثمانون من مئة")
    assert CASES[57].values == ("1 1/2", "واحد و نصف")


@pytest.mark.parametrize("written, spoken", CASES)
def test_original(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written, spoken", CASES)
def test_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text == spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert spoken[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        written, spoken = param.values
        assert normalizer.verbalize(normalizer.tag(written)) == spoken


@pytest.mark.parametrize("written, expected", [
    ("9837", [("cardinal", {"integer": "9837"})]),
    ("-5", [("cardinal", {"negative": "-", "integer": "5"})]),
    ("7,99", [("decimal", {"integer_part": "7", "fractional_part": "99"})]),
    ("1 1/2", [("fraction", {"integer_part": "1", "numerator": "1", "denominator": "2"})]),
    ("1½", [("fraction", {"integer_part": "1", "vulgar": "½"})]),
    ("$7.05", [("money", {"currency_maj": "$", "integer_part": "7", "fractional_part": "05"})]),
    ("5د.ك", [("money", {"integer_part": "5", "currency_maj": "د.ك"})]),
    ("%6.88", [("measure", {"units": "%", "integer_part": "6", "fractional_part": "88"})]),
])
def test_raw_field_tags(normalizer, written, expected):
    assert parse(normalizer, written) == expected


@pytest.mark.parametrize("written, spoken", [
    ("-5", "سالب خمسة"),
    ("3100", "ثلاثة اَلاف ومئة"),
    ("½", "نصف"),
    ("1½", "واحد و نصف"),
    ("$1", "دولار واحد"),
    ("لدي 3 كتب", "لدي ثلاثة كتب"),
    ("5 km", "خمسة km"),
    ("5-kg", "خمسة kg"),
])
def test_hand_written(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written", ["3006", "6.88", "1 1/2", "$7.50", "%7", "5د.ك"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)


def test_escaped_word_round_trips(normalizer):
    written = 'مسار\\a"b'
    assert parse(normalizer, written) == [("word", {"value": written})]
    assert normalizer.normalize(written) == written


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("written", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, written):
    tagged = normalizer.tag(written)
    assert parse(normalizer, written) == [("word", {"value": written})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(written, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
