# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from itn.french.inverse_normalizer import InverseNormalizer


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


def test_original_fixture_count_and_trailing_spaces():
    assert len(CASES) == 307
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
    "cent cinquante et un", "premier janvier", "deux euros et vingt centimes", "un et demi", "midi moins le quart",
])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, kind, members", [
    ("moins vingt-trois", "cardinal", {"negative": "moins", "integer": "vingt-trois"}),
    ("cent cinquante et un", "cardinal", {"integer": "cent cinquante et un"}),
    ("vingt-troisièmes", "ordinal", {"integer": "vingt-troisièmes"}),
    ("dix-neuvième siècle", "ordinal", {"integer": "dix-neuvième", "noun": "siècle"}),
    ("moins un virgule deux six", "decimal", {
        "negative": "moins", "integer_part": "un", "fractional_part": "deux six",
    }),
    ("quatre-cent-soixante-millions", "decimal", {"integer_part": "quatre-cent-soixante", "quantity": "millions"}),
    ("quatre et deux quatrièmes", "fraction", {
        "integer_part": "quatre", "numerator": "deux", "denominator": "quatrièmes",
    }),
    ("demi", "fraction", {"denominator": "demi"}),
    ("premier janvier deux-mille-treize", "date", {"day": "premier", "month": "janvier", "year": "deux-mille-treize"}),
    ("midi moins le quart", "time", {"hours": "midi", "minutes": "moins le quart"}),
    ("huit heures du soir", "time", {"hours": "huit heures", "suffix": "du soir"}),
    ("deux euros et vingt centimes", "money", {
        "integer_part": "deux", "currency": "euros", "fractional_part": "vingt", "currency_minor": "centimes",
    }),
    ("trois pence", "money", {"fractional_part": "trois", "currency": "pence"}),
    ("deux-millions de dollars", "money", {"integer_part": "deux", "quantity": "millions", "currency": "de dollars"}),
    ("moins soixante-six kilogrammes", "measure", {
        "negative": "moins", "integer": "soixante-six", "units": "kilogrammes",
    }),
    ("double neuf douze trente-deux trente trente", "telephone", {
        "number_part": "double neuf douze trente-deux trente trente",
    }),
    ("a b c arobase g mail point com", "electronic", {"username": "a b c", "domain": "g mail point com"}),
    ("mademoiselle", "whitelist", {"value": "mademoiselle"}),
])
def test_raw_fields(normalizer, spoken, kind, members):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == [kind]
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, written", [
    ("onze heures vingt-quatre", "11 h 24"),
    ("vingt et une heures", "21 h"),
    ("premier siècle", "Iᵉʳ siècle"),
    ("deux-cent neuf", "200 neuf"),
    ("« vingt »", "« 20 »"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


def test_escaped_word_round_trips(normalizer):
    spoken = 'chemin\\a"b'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": spoken}]
    assert normalizer.normalize(spoken) == spoken


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
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
