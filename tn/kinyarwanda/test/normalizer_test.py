# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from pathlib import Path

import pytest

from tn.kinyarwanda.normalizer import Normalizer

PACKAGE = Path(__file__).resolve().parent.parent


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
    assert len(CASES) == 100
    # A whole hour keeps NeMo's trailing space from the "00" minute reading.
    assert CASES[57].values == ("1:00", "saa saba ")
    # The time and whitelist fixtures have no trailing newline.
    assert CASES[70].values == ("12:31", "saa sita n'iminota mirongo itatu n'umwe")
    assert CASES[73].values == ("television", "televiziyo")
    assert CASES[74].values == ("", "")
    assert CASES[75].values == (" ", "")


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


@pytest.mark.parametrize("written, kind, members, spoken", [
    ("123", "cardinal", {"integer": "123"}, "ijana na makumyabiri na gatatu"),
    ("0", "cardinal", {"integer": "0"}, "zeru"),
    ("002", "cardinal", {"digits": "002"}, "zeru zeru kabiri"),
    ("2,3,4,5", "cardinal", {"digits": "2,3,4,5"}, "kabiri gatatu kane gatanu"),
    ("1000", "cardinal", {"integer": "1000"}, "igihumbi"),
    ("1001000", "cardinal", {"integer": "1001000"}, "miliyoni n'igihumbi"),
    ("21000000", "cardinal", {"integer": "21000000"}, "miliyoni makumyabiri n'imwe"),
    ("1000000000000", "cardinal", {"digits": "1000000000000"},
     "rimwe zeru zeru zeru zeru zeru zeru zeru zeru zeru zeru zeru zeru"),
    ("3:30", "time", {"hours": "3", "minutes": "30"}, "saa cyenda n'iminota mirongo itatu"),
    ("11:00", "time", {"hours": "11", "minutes": "00"}, "saa tanu "),
    ("maroc", "whitelist", {"value": "maroc"}, "maroki"),
    ("la crosse", "whitelist", {"value": "la crosse"}, "la korosi"),
    ("ADN", "whitelist", {"value": "ADN"}, "ade eni"),
    ("umwana", "word", {"value": "umwana"}, "umwana"),
    ("13:00", "word", {"value": "13:00"}, "13:00"),
])
def test_raw_fields(normalizer, written, kind, members, spoken):
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == kind
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == spoken
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written, spoken", [
    ("(5)", "(gatanu)"),
    ("5,", "gatanu,"),
    ("12:31.", "saa sita n'iminota mirongo itatu n'umwe."),
    ("Chris na 3:05", "kiris na saa cyenda n'iminota itanu"),
    ("2,345", "kabiri gatatu kane gatanu"),
    ("X!", "X!"),
    ("Nzaza 1:00 ejo", "Nzaza saa saba ejo"),
    ("11:00, 12:00", "saa tanu , saa sita "),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written, spoken, punctuation", [
    ("(5)", "(gatanu)", ["(", ")"]),
    ("5,", "gatanu,", [","]),
    ("5 !", "gatanu !", [" !"]),
    ("( 5 )", "( gatanu )", ["( ", " )"]),
])
def test_cardinal_punctuation_spacing(normalizer, written, spoken, punctuation):
    assert normalizer.normalize(written) == spoken
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.members for token in parser.tokens if token.name == "cardinal"] == [{"integer": "5"}]
    assert [token.members for token in parser.tokens if token.name == "punctuation"] == [
        {"value": value} for value in punctuation
    ]
    assert normalizer.verbalize(tagged) == spoken
    mapped = normalizer.normalize_with_mapping(written, include_identity=True)
    assert mapped.output_text == spoken
    assert [mapping.input_text for mapping in mapped.mappings if mapping.token_type == "punctuation"] == punctuation
    for mapping in mapped.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert spoken[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("written", ["Nzaza 1:00 ejo", "11:00, 12:00", "1:00"])
def test_whole_hour_mapping_is_consistent(normalizer, written):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text == normalizer.normalize(written)
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("written", ["123", "2,3,4,5", "12:31", "maroc", "(5)"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)
    for result in mapped:
        for mapping in result.mappings:
            assert written[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_escaped_word_round_trips(normalizer):
    written = 'umwana\\a"b'
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


def test_standalone_from_itn():
    itn_import = re.compile(r"^\s*(from|import)\s+itn\.kinyarwanda\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not itn_import.search(path.read_text(encoding="utf-8")), path
