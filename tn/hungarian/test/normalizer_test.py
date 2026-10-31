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

from tn.hungarian.normalizer import Normalizer


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
    # 204 newline-terminated lines plus the unterminated last electronic case.
    assert len(CASES) == 205
    assert CASES[0].values == ("2", "kettő")
    assert CASES[153].values == ("", "")


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
    ("1 000", "cardinal", {"integer": "1 000"}, "ezer"),
    ("-12", "cardinal", {"negative": "-", "integer": "12"}, "mínusz tizenkettő"),
    ("131 111.", "ordinal", {"integer": "131 111"}, "százharmincegyezer-száztizenegyedik"),
    ("-11,4006 milliárd", "decimal", {
        "negative": "-", "integer_part": "11", "fractional_part": "4006", "quantity": "milliárd",
    }, "mínusz tizenegy egész négyezer-hat tízezred milliárd"),
    ("1 mrd", "decimal", {"integer_part": "1", "quantity": "mrd"}, "egy milliárd"),
    ("21/3443", "fraction", {"numerator": "21", "denominator": "3443"},
     "huszonegy háromezer-négyszáznegyvenharmad"),
    ("⅔", "fraction", {"symbol": "⅔"}, "kettő harmad"),
    ("1 1/2", "fraction", {"integer_part": "1", "numerator": "1", "denominator": "2"}, "egy és egy fél"),
    ("3mm-ben", "measure", {"integer": "3", "units": "mm-ben"}, "három milliméterben"),
    ("2,3 mm", "measure", {"integer_part": "2", "fractional_part": "3", "units": "mm"},
     "kettő egész három tized milliméter"),
    ("$1,01", "money", {"currency": "$", "integer_part": "1", "minor": "01"}, "egy dollár egy cent"),
    ("$0,50", "money", {"currency": "$", "integer_part": "0", "minor": "50"}, "ötven cent"),
    ("$2,00", "money", {"currency": "$", "integer_part": "2", "fractional_part": "00"}, "kettő dollár"),
    ("1,32 GBP-t", "money", {"integer_part": "1", "fractional_part": "32", "currency": "GBP-t"},
     "egy egész harminckettő század angol fontot"),
    ("1 mrd Ft", "money", {"integer_part": "1", "quantity": "mrd", "currency": "Ft"}, "egy milliárd forint"),
    ("2010. ápr. 1-jén", "date", {"year": "2010", "month": "ápr", "day": "1-jén"},
     "kettőezer-tíz április elsején"),
    ("2010-04-01", "date", {"year": "2010", "month": "04", "day": "01"}, "kettőezer-tíz április elseje"),
    ("1.4.2010", "date", {"day": "1", "month": "4", "year": "2010"}, "kettőezer-tíz április elseje"),
    ("08:05:00", "time", {"hours": "08", "minutes": "05", "seconds": "00"}, "nyolc óra öt perc"),
    ("8 ó 5 p-cel", "time", {"hours": "8", "minutes": "5", "suffix": "p-cel"}, "nyolc óra öt perccel"),
    ("00:00 cet-t", "time", {"hours": "00", "minutes": "00", "zone": "cet-t"}, "nulla óra c e t-t"),
    ("tel.: +36-80/555-111", "telephone", {"prompt": "tel.", "country_code": "+36", "number_part": "80/555-111"},
     "telefon plusz harminchat nyolcvan ötszázötvenöt száztizenegy"),
    ("193.1.89.60", "telephone", {"number_part": "193.1.89.60"},
     "százkilencvenhárom pont egy pont nyolcvankilenc pont hatvan"),
    ("abc@def.hu", "electronic", {"username": "abc", "domain": "def.hu"}, "a b c kukac d e f pont hu"),
    ("https://www.nvidia.com", "electronic", {"protocol": "https://www.", "domain": "nvidia.com"},
     "h t t p s kettőspont perjel perjel w w w pont nvidia pont com"),
    ("@wezyr1986", "electronic", {"username": "wezyr1986"}, "kukac w e z y r egy kilenc nyolc hat"),
    ("tv.-nyel", "whitelist", {"value": "tv.-nyel"}, "törvénnyel"),
    ("és tsai.", "whitelist", {"value": "és tsai."}, "és társai"),
    ("köszönöm", "word", {"value": "köszönöm"}, "köszönöm"),
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
    ("(5)", "( öt )"),
    ("5,", "öt ,"),
    ("1 034 068", "millió-harmincnégyezer-hatvannyolc"),
    ("Április 5.", "április ötödike"),
    ("2010. május", "kettőezer-tíz május"),
    ("8 óra cet", "nyolc óra c e t"),
    ("8 ó 5 p 3 mp", "nyolc óra öt perc három másodperc"),
    ("5%", "öt százalék"),
    ("$1,5", "egy dollár ötven cent"),
    ("$1,234", "egy egész kétszázharmincnégy ezred dollár"),
    ("2 USD-t", "kettő amerikai dollárt"),
    ("0036 1 441 4000", "nulla nulla harminchat egy négyszáznegyvenegy négyezer"),
    ("x@y.com", "x kukac y pont com"),
    ("Bp.-en", "Budapesten"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written", ["1/2", "15:52", "$1,01", "131 111.", "2010. 04. 1-jén", "tv.-nyel"])
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
    written = 'út\\a"b'
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
