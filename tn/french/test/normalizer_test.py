# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.french.normalizer import Normalizer


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
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def test_original_fixture_count():
    assert len(CASES) == 117


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


@pytest.mark.parametrize("written", ["le 02.03.2003", "1 1/2", "-60", "18,2 milliards", "21e", "Mᵐᵉ"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)


@pytest.mark.parametrize("written, kind, members", [
    ("23", "cardinal", {"integer": "23"}),
    ("-6 492 000", "cardinal", {"negative": "-", "integer": "6 492 000"}),
    ("21e", "ordinal", {"integer": "21", "morphosyntactic_features": "e"}),
    ("-11,406 millions", "decimal", {
        "negative": "-", "integer_part": "11", "fractional_part": "406", "quantity": "millions",
    }),
    ("-4 1/3", "fraction", {"negative": "-", "integer_part": "4", "numerator": "1", "denominator": "3"}),
    ("le 02.03.2003", "date", {"day": "le 02", "month": "03", "year": "2003"}),
    ("les 17-18-19 juin", "date", {"day": "les 17-18-19", "month": "juin"}),
    ("80s", "date", {"year": "80s"}),
    ("Mᵐᵉ", "whitelist", {"value": "Mᵐᵉ"}),
])
def test_raw_fields(normalizer, written, kind, members):
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == [kind]
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(written)


@pytest.mark.parametrize("written, spoken", [
    ("01.03.2003", "premier mars deux mille trois"),
    ("17 février 2003", "dix-sept février deux mille trois"),
    ("5 1/2", "cinq et demi"),
    ("4er", "4er"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


def test_escaped_word_round_trips(normalizer):
    written = 'chemin\\a"b'
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
