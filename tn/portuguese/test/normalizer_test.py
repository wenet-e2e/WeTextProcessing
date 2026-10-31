# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.portuguese.normalizer import Normalizer


def original_cases():
    """Loads the NeMo fixtures; extra "~" columns are alternative correct readings."""
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                written, *spoken = line.split("~")
                cases.append(pytest.param(written, tuple(spoken), id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def test_original_fixture_count():
    assert len(CASES) == 359
    assert sum(len(case.values[1]) > 1 for case in CASES) == 3


@pytest.mark.parametrize("written, spoken", CASES)
def test_original(normalizer, written, spoken):
    assert normalizer.normalize(written) in spoken


@pytest.mark.parametrize("written, spoken", CASES)
def test_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text in spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        written, spoken = param.values
        assert normalizer.verbalize(normalizer.tag(written)) == normalizer.normalize(written)
        assert normalizer.normalize(written) in spoken


@pytest.mark.parametrize("written", ["15/03/2024", "1/2", "R$ 12,05", "14:30", "1 kg", "test@gmail.com"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)


@pytest.mark.parametrize("written, kind, members", [
    ("23", "cardinal", {"integer": "23"}),
    ("1.000.000", "cardinal", {"integer": "1.000.000"}),
    ("-1,26", "decimal", {"negative": "-", "integer_part": "1", "fractional_part": "26"}),
    ("1,5 milhões", "decimal", {"integer_part": "1", "fractional_part": "5", "quantity": "milhões"}),
    ("11ª", "ordinal", {"integer": "11", "suffix": "ª"}),
    ("3 2/3", "fraction", {"integer_part": "3", "numerator": "2", "denominator": "3"}),
    ("2024-03-15", "date", {"year": "2024", "month": "03", "day": "15"}),
    ("10 de Dezembro de 1999", "date", {"day": "10", "month": "Dezembro", "year": "1999"}),
    ("15h da tarde", "time", {"hours": "15", "suffix": "da tarde"}),
    ("09:00:31", "time", {"hours": "09", "minutes": "00", "seconds": "31"}),
    ("2 h", "measure", {"integer": "2", "units": "h"}),
    ("R$ 12,05", "money", {"currency_maj": "R$", "integer_part": "12", "fractional_part": "05"}),
    ("(11) 99999-8888", "telephone", {"number_part": "(11) 99999-8888"}),
    ("test@gmail.com", "electronic", {"username": "test", "domain": "gmail.com"}),
    ("%", "whitelist", {"value": "%"}),
])
def test_raw_fields(normalizer, written, kind, members):
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == [kind]
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(written)


@pytest.mark.parametrize("written, spoken", [
    ("Tenho 5 anos.", "Tenho cinco anos."),
    ("ligue para 555-1234", "ligue para cinco cinco cinco um dois três quatro"),
    ("R$ 1000000,50", "um milhão de reais e cinquenta centavos"),
    ("13:00 da tarde", "uma hora da tarde"),
    ("1 1/2 kg", "um e um meio quilos"),
    ("3/2", "três meios"),
    ("A.B.C", "ABC"),
    ("1.º", "primeiro"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


def test_escaped_word_round_trips(normalizer):
    written = 'caminho\\a"b'
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
