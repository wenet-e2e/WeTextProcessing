# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from itn.hindi.inverse_normalizer import InverseNormalizer


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                spoken, written = line.split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def inverse_normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_original_fixture_count():
    assert len(CASES) == 352


@pytest.mark.parametrize("spoken, written", CASES)
def test_original(inverse_normalizer, spoken, written):
    assert inverse_normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", CASES)
def test_original_mapping(inverse_normalizer, spoken, written):
    result = inverse_normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken", ["तेईस", "दसवां", "एक दशमलव पाँच", "सवा तीन बजे", "बारह रुपये", "डॉक्टर"])
def test_nbest_mapping_synchronization(inverse_normalizer, spoken):
    outputs = inverse_normalizer.normalize(spoken, nbest=3)
    mapped = inverse_normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == inverse_normalizer.normalize(spoken)


def test_tag_then_verbalize_matches_normalize(inverse_normalizer):
    for param in CASES:
        spoken, written = param.values
        assert inverse_normalizer.verbalize(inverse_normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken, name, members", [
    # Like upstream, a bare number of up to four digits is cheapest as a year.
    ("तेईस", "date", {"value": "तेईस"}),
    ("एक लाख", "cardinal", {"integer": "एक लाख"}),
    ("ऋणात्मक तेईस", "cardinal", {"negative": "ऋणात्मक", "integer": "तेईस"}),
    ("दसवां", "ordinal", {"integer": "दसवां"}),
    ("ऋण एक दशमलव दो छह", "decimal", {"negative": "ऋण", "integer_part": "एक", "fractional_part": "दो छह"}),
    ("बारह रुपये पचास पैसे", "money",
     {"currency": "रुपये", "value": "बारह", "fractional_part": "पचास", "minor_currency": "पैसे"}),
    ("दो बजकर ग्यारह मिनट", "time", {"value": "दो बजकर ग्यारह मिनट"}),
])
def test_raw_field_tags(inverse_normalizer, spoken, name, members):
    parser = inverse_normalizer.token_parser()
    parser.parse(inverse_normalizer.tag(spoken))
    assert [token.name for token in parser.tokens] == [name]
    assert parser.tokens[0].members == members


def test_hand_written_conversions(inverse_normalizer):
    # Money fields are reordered so the currency symbol is verbalized first.
    assert inverse_normalizer.normalize("बारह रुपये पचास पैसे") == "₹१२.५०"
    assert inverse_normalizer.normalize("एक लाख") == "१०००००"
    assert inverse_normalizer.normalize("ऋणात्मक तेईस") == "-२३"


def test_escaped_word_round_trips(inverse_normalizer):
    spoken = 'पथ\\a"b'
    parser = inverse_normalizer.token_parser()
    parser.parse(inverse_normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": spoken}]
    assert inverse_normalizer.normalize(spoken) == spoken


def test_empty_input(inverse_normalizer):
    assert inverse_normalizer.normalize("") == ""
    assert inverse_normalizer.normalize("", nbest=3) == [""]
    assert inverse_normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(inverse_normalizer, spoken):
    tagged = inverse_normalizer.tag(spoken)
    parser = inverse_normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": spoken}
    assert inverse_normalizer.verbalize(tagged) == ""
    assert inverse_normalizer.normalize(spoken, nbest=3) == [""]
    mapped = inverse_normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
