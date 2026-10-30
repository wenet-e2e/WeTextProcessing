# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from tn.korean.normalizer import Normalizer


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                written, spoken = line.split("~")
                cases.append(pytest.param(written, spoken, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def normalizer():
    return Normalizer(cache_dir=False)


def test_original_fixture_count_and_trailing_spaces():
    assert len(CASES) == 395
    assert CASES[68].values == ("2024년", "이천이십사년 ")


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


@pytest.mark.parametrize("written", ["2024년", "1/2", "$10", "10:30", "12.5kg", "abc@gmail.com"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        written, spoken = param.values
        assert normalizer.verbalize(normalizer.tag(written)) == spoken


def test_escaped_word_round_trips(normalizer):
    written = '경로\\a"b'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert [token.members for token in parser.tokens] == [{"value": written}]
    assert normalizer.normalize(written) == written


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("written", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, written):
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": written}
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(written, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
