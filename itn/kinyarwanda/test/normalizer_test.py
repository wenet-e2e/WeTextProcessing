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

from itn.kinyarwanda.inverse_normalizer import InverseNormalizer

PACKAGE = Path(__file__).resolve().parent.parent
# NeMo has no Kinyarwanda ITN fixtures; the TN fixtures are read reversed.
TN_DATA = PACKAGE.parent.parent / "tn" / "kinyarwanda" / "test" / "data"

# Reversed TN pairs that cannot hold, with the output ITN gives instead.
EXCLUSIONS = {
    # TN deletes the commas of a digit list; the separators are not recoverable.
    "kabiri gatatu kane gatanu": "2 3 4 5",
    # Transliterations are the written Kinyarwanda words and are not inverted.
    "maroki": "maroki",
    "televiziyo": "televiziyo",
    # TN strips the surrounding whitespace that the written side keeps.
    "no": "no",
    "x": "x",
}


def reversed_cases():
    candidates = {}
    for path in sorted(TN_DATA.glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.endswith("\n"):
                    line = line[:-1]
                written, spoken = line.split("~")
                candidates.setdefault(spoken, []).append(written)
    return candidates


REVERSED = reversed_cases()
CASES = [pytest.param(spoken, written, id=spoken or "empty")
         for spoken, written in REVERSED.items() if spoken not in EXCLUSIONS]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_reversed_fixture_count():
    assert sum(len(written) for written in REVERSED.values()) == 100
    assert len(REVERSED) == 98
    assert len(CASES) == 93
    # Whole-hour readings end with a space and whitespace-only input reads as "".
    assert REVERSED["saa saba "] == ["1:00"]
    assert REVERSED[""] == ["", " "]
    assert REVERSED["rimwe"] == ["1", "1"]


@pytest.mark.parametrize("spoken, written", CASES)
def test_reversed(normalizer, spoken, written):
    assert normalizer.normalize(spoken) in written


@pytest.mark.parametrize("spoken, written", CASES)
def test_reversed_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text in written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken, output", sorted(EXCLUSIONS.items()))
def test_exclusions(normalizer, spoken, output):
    assert spoken in REVERSED
    assert output not in REVERSED[spoken]
    assert normalizer.normalize(spoken) == output


def test_tag_then_verbalize_matches_normalize(normalizer):
    for spoken in list(REVERSED) + list(EXCLUSIONS):
        assert normalizer.verbalize(normalizer.tag(spoken)) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, kind, members, written", [
    ("ijana na makumyabiri na gatatu", "cardinal", {"integer": "ijana na makumyabiri na gatatu"}, "123"),
    ("zeru", "cardinal", {"integer": "zeru"}, "0"),
    ("zeru zeru kabiri", "cardinal", {"digits": "zeru zeru kabiri"}, "002"),
    ("igihumbi", "cardinal", {"integer": "igihumbi"}, "1000"),
    ("miliyoni n'igihumbi", "cardinal", {"integer": "miliyoni n'igihumbi"}, "1001000"),
    ("miliyoni imwe", "cardinal", {"integer": "miliyoni imwe"}, "1000000"),
    ("tiriyoni imwe", "cardinal", {"integer": "tiriyoni imwe"}, "1000000000"),
    ("miliyoni makumyabiri n'imwe", "cardinal", {"integer": "miliyoni makumyabiri n'imwe"}, "21000000"),
    ("saa cyenda n'iminota mirongo itatu", "time", {"hours": "cyenda", "minutes": "iminota mirongo itatu"},
     "3:30"),
    ("saa munani n'umunota umwe", "time", {"hours": "munani", "minutes": "umunota umwe"}, "2:01"),
    ("saa cumi n'imwe", "time", {"hours": "cumi n'imwe"}, "5:00"),
    ("saa tanu", "time", {"hours": "tanu"}, "11:00"),
    ("bibisi", "whitelist", {"value": "bibisi"}, "BBC"),
    ("ade eni", "whitelist", {"value": "ade eni"}, "ADN"),
    ("wawawa", "whitelist", {"value": "wawawa"}, "www"),
    ("umwana", "word", {"value": "umwana"}, "umwana"),
    ("televiziyo", "word", {"value": "televiziyo"}, "televiziyo"),
])
def test_raw_fields(normalizer, spoken, kind, members, written):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == kind
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == written
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", [
    ("(gatanu)", "( 5 )"),
    ("gatanu,", "5 ,"),
    ("saa saba.", "1:00 ."),
    ("ndi na makumyabiri", "ndi na 20"),
    ("vi o aye na bibisi", "VOA na BBC"),
    # TN writes both 130000 and 100030 this way; the larger scale keeps the parts.
    ("ibihumbi ijana na mirongo itatu", "130000"),
    ("ibihumbi ijana na makumyabiri na gatatu", "120003"),
    ("rimwe kabiri", "1 2"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


def test_ambiguous_reading_keeps_alternative(normalizer):
    assert normalizer.normalize("ibihumbi ijana na mirongo itatu", nbest=2) == ["130000", "100030"]


@pytest.mark.parametrize("spoken", ["ijana na makumyabiri na gatatu", "saa sita n'iminota icyenda", "bibisi",
                                    "zeru zeru kabiri", "(gatanu)"])
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


def test_escaped_word_round_trips(normalizer):
    spoken = 'umwana\\a"b'
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


def test_standalone_from_tn():
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.kinyarwanda\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
