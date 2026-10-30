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

from itn.portuguese.inverse_normalizer import InverseNormalizer


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
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_original_fixture_count_and_spaces():
    assert len(CASES) == 281
    assert ("", "") in [case.values for case in CASES]
    assert ("x ", "x") in [case.values for case in CASES]


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


@pytest.mark.parametrize("spoken", [
    "vinte e três", "um vírgula dois seis", "doze dólares e cinco centavos",
    "às quinze pras duas da tarde", "dez do oito de dois mil", "a b c arroba a b c ponto com",
])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)
    for result in mapped:
        for mapping in result.mappings:
            assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken, kind, members", [
    ("vinte e três", "cardinal", {"integer": "vinte e três"}),
    ("menos sessenta", "cardinal", {"negative": "menos", "integer": "sessenta"}),
    ("menos um vírgula dois seis", "decimal", {
        "negative": "menos", "integer_part": "um", "point": "vírgula", "fractional_part": "dois seis",
    }),
    ("dois milhões", "decimal", {"integer_part": "dois", "quantity": "milhões"}),
    ("vigésima primeira", "ordinal", {"integer": "vigésima primeira"}),
    ("doze dólares e cinco centavos", "money", {
        "integer_part": "doze", "currency": "dólares", "cents": "cinco", "currency_min": "centavos",
    }),
    ("vinte centavos", "money", {"cents": "vinte", "currency_min": "centavos"}),
    ("dois quilos", "measure", {"integer": "dois", "units": "quilos"}),
    ("dez do oito de dois mil", "date", {"day": "dez", "month": "oito", "year": "dois mil"}),
    ("às quinze pras duas da tarde", "time", {
        "prefix": "às", "minutes": "quinze", "to": "pras", "hours": "duas", "suffix": "da tarde",
    }),
    ("um dois três quatro cinco seis sete oito", "telephone", {
        "number_part": "um dois três quatro cinco seis sete oito",
    }),
    ("c d f arroba a b c ponto e d u", "electronic", {"username": "c d f", "domain": "a b c ponto e d u"}),
    ("sexta feira", "whitelist", {"value": "sexta feira"}),
    ("aaa", "word", {"value": "aaa"}),
])
def test_raw_fields(normalizer, spoken, kind, members):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == [kind]
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, written", [
    ("vivo em uma casa", "vivo em uma casa"),
    ("segunda guerra mundial", "segunda guerra mundial"),
    ("cinquenta, dez.", "50 , 10 ."),
    ("um bilhão e mil cento e um", "1000001101"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


def test_escaped_word_round_trips(normalizer):
    spoken = 'caminho\\a"b dez'
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == 'caminho\\a"b 10'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": 'caminho\\a"b'}, {"integer": "dez"}]


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, spoken):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": spoken}
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
