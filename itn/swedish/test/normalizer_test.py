# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.swedish.inverse_normalizer import InverseNormalizer


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_original_fixture_count_and_trailing_spaces():
    assert len(CASES) == 135
    assert ("x ", "x") in [case.values for case in CASES]


@pytest.mark.parametrize("spoken, written", CASES)
def test_original(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written
    assert normalizer.verbalize(normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken, written", CASES)
def test_original_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken", [
    "tjugo en", "tre komma fem miljoner", "kvart i tolv", "andra januari tjugotjugotvå", "fyra femtedelar",
])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, members", [
    ("minus tjugo en", [{"negative": "minus", "integer": "tjugo en"}]),
    ("hundratjugotredje", [{"integer": "hundratjugotredje"}]),
    ("tre komma fem miljon", [{"integer_part": "tre", "fractional_part": "fem", "quantity": "miljon"}]),
    ("två och halv", [{"integer_part": "två", "denominator": "halv"}]),
    ("tre i kvart i tolv", [{"minutes": "tre i kvart i", "hours": "tolv"}]),
    ("klockan tolv eftermiddag c e t", [{"prefix": "klockan", "hours": "tolv", "suffix": "eftermiddag",
                                         "zone": "c e t"}]),
    ("andra januari tjugotjugotvå före Kristus", [{"day": "andra", "month": "januari", "year": "tjugotjugotvå",
                                                   "era": "före Kristus"}]),
    ("a b c snabel-a g mail punkt com", [{"username": "a b c", "domain": "g mail punkt com"}]),
])
def test_raw_fields(normalizer, spoken, members):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == members


def test_reordered_fields(normalizer):
    tagged = 'date { day: "andra" month: "januari" year: "tjugotjugotvå" }'
    assert normalizer.verbalize(tagged) == "2022-01-02"
    assert normalizer.normalize("halv tolv") == "11:30"


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, spoken):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": spoken}
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)


def test_escaped_word_round_trips(normalizer):
    spoken = 'gränd\\väg"中文'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": spoken}]
    assert normalizer.normalize(spoken) == spoken


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""
