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

from itn.hungarian.inverse_normalizer import InverseNormalizer

# NeMo has no Hungarian ITN fixtures: the TN fixtures are read in reverse.
TN_DATA = Path(__file__).resolve().parents[3] / "tn" / "hungarian" / "test" / "data"

# Reversed TN pairs that cannot hold, each with the reason.
EXCLUDED = {
    "kettőmillió": "only the dot-grouped '2.000.000' is listed; ITN groups with spaces like '1 034 068'",
    "milliárd": "only the dot-grouped '1.000.000.000' is listed; ITN groups with spaces",
    "kettőmilliárd": "only the dot-grouped '2.000.000.000' is listed; ITN groups with spaces",
    "százharmincegyezer-száztizenegyedik": "only '131 111.' is listed; ITN writes ordinals ungrouped like '455456.'",
    "mínusz tizenegy egész négyezer-hat tízezred milliárd":
        "only '-11,4006 milliárd' is listed; ITN abbreviates milliárd as in '$1 mrd' and '1 mrd GBP-t'",
    "kettő egész három tized milliméter": "only '2,3 mm' is listed; ITN attaches units as in '3mm' and '12kg'",
    "kettőezer-tíz április elsején": "only numeric months ('2010. 04. 1-jén') are listed; ITN keeps month names",
    "április elsején": "only numeric months ('04. 1-jén') are listed; ITN keeps month names",
    "plusz harminchat egy négyszáznegyvenegy négyezer":
        "the listed '+ (36) 1 441-4000' layout is not recoverable from speech",
    "plusz harminchat nyolcvan ötszázötvenöt száztizenegy":
        "'+36-80/555-111' conflicts with the space-separated '+36 1 4888 700' layout",
    "telefon plusz harminchat nyolcvan ötszázötvenöt száztizenegy":
        "'tel.: +36-80/555-111' uses the dash/slash layout and a colon",
    "telefon nulla hat negyven negyvenkilenc négyezer-kilencszáznegyvenkilenc":
        "'tel. 06-40-49-4949' uses the dash layout; ITN writes '06 40 49 4949'",
}


def reversed_cases():
    candidates = {}
    lines = 0
    for path in sorted(TN_DATA.glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.endswith("\n"):
                    line = line[:-1]
                written, spoken = line.split("~")
                candidates.setdefault(spoken, []).append(written)
                lines += 1
    return lines, candidates


LINES, CANDIDATES = reversed_cases()
CASES = [pytest.param(spoken, written, id=spoken or "empty")
         for spoken, written in CANDIDATES.items() if spoken not in EXCLUDED]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_reversed_fixture_count():
    assert LINES == 205
    assert len(CANDIDATES) == 178
    assert set(EXCLUDED) <= set(CANDIDATES)
    assert len(CASES) == 166
    assert CANDIDATES["ezer"] == ["1000", "1 000", "1.000"]


@pytest.mark.parametrize("spoken, written", CASES)
def test_reversed(normalizer, spoken, written):
    assert normalizer.normalize(spoken) in written


@pytest.mark.parametrize("spoken", sorted(EXCLUDED))
def test_excluded_pairs_still_fail(normalizer, spoken):
    # Keeps the exclusion list minimal: an exclusion that starts passing must be removed.
    assert normalizer.normalize(spoken) not in CANDIDATES[spoken]


@pytest.mark.parametrize("spoken, written", CASES)
def test_reversed_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text in written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for spoken in CANDIDATES:
        assert normalizer.verbalize(normalizer.tag(spoken)) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, kind, members, written", [
    ("kétezer tíz", "cardinal", {"integer": "kétezer tíz"}, "2010"),
    ("tízezer", "cardinal", {"integer": "tízezer"}, "10 000"),
    ("mínusz öt", "cardinal", {"negative": "mínusz", "integer": "öt"}, "-5"),
    ("huszadik", "ordinal", {"integer": "huszadik"}, "20."),
    ("tizenegy egész négyezer-hat százezred", "decimal",
     {"integer_part": "tizenegy", "fractional_part": "négyezer-hat százezred"}, "11,04006"),
    ("kettő egész öt tized millió", "decimal",
     {"integer_part": "kettő", "fractional_part": "öt tized", "quantity": "millió"}, "2,5 millió"),
    ("egy milliárd", "decimal", {"integer_part": "egy", "quantity": "milliárd"}, "1 mrd"),
    ("három és egy negyed", "fraction", {"integer_part": "három", "numerator": "egy", "denominator": "negyed"},
     "3 1/4"),
    ("kettő harmad", "fraction", {"numerator": "kettő", "denominator": "harmad"}, "2/3"),
    ("három milliméterben", "measure", {"integer": "három", "units": "milliméterben"}, "3mm-ben"),
    ("tíz kilométer per óra", "measure", {"integer": "tíz", "units": "kilométer per óra"}, "10km/ó"),
    ("öt százalék", "measure", {"integer": "öt", "units": "százalék"}, "5%"),
    ("egy dollár egy cent", "money",
     {"integer_part": "egy", "currency": "dollár", "minor": "egy", "minor_currency": "cent"}, "$1,01"),
    ("egy milliárd angol fontot", "money",
     {"integer_part": "egy", "quantity": "milliárd", "currency": "angol fontot"}, "1 mrd GBP-t"),
    ("száz forint ötven fillér", "money",
     {"integer_part": "száz", "currency": "forint", "minor": "ötven", "minor_currency": "fillér"}, "100,50 Ft"),
    ("ezerkilencszáznegyvennyolc március tizenötödike", "date",
     {"year": "ezerkilencszáznegyvennyolc", "month": "március", "day": "tizenötödike"}, "1948. március 15."),
    ("május elsején", "date", {"month": "május", "day": "elsején"}, "május 1-jén"),
    ("kettőezer-huszonhat október", "date", {"year": "kettőezer-huszonhat", "month": "október"}, "2026. október"),
    ("tizenkét óra harminc perc", "time", {"hours": "tizenkét", "minutes": "harminc"}, "12:30"),
    ("nyolc óra öt másodperc", "time", {"hours": "nyolc", "seconds": "öt"}, "08:00:05"),
    ("nyolc órakor", "time", {"hours": "nyolc", "suffix": "órakor"}, "8 ó-kor"),
    ("nulla óra c e t-t", "time", {"hours": "nulla", "zone": "c e t-t"}, "00:00 cet-t"),
    ("plusz harminchat húsz ötszázötvenöt hatvanhat hatvanhat", "telephone",
     {"country_code": "plusz harminchat", "number_part": "húsz ötszázötvenöt hatvanhat hatvanhat"},
     "+36 20 555 66 66"),
    ("nulla hat egy négyszáznegyvenegy négyezer", "telephone",
     {"number_part": "nulla hat egy négyszáznegyvenegy négyezer"}, "06 1 441 4000"),
    ("x kukac y pont com", "electronic", {"username": "x", "domain": "y pont com"}, "x@y.com"),
    ("index pont hu", "electronic", {"domain": "index pont hu"}, "index.hu"),
    ("kukac j e n s e n pont m e", "electronic", {"username": "j e n s e n pont m e"}, "@jensen.me"),
    ("s a többi", "whitelist", {"value": "s a többi"}, "stb."),
    ("törvénnyel", "whitelist", {"value": "törvénnyel"}, "tv.-nyel"),
    ("köszönöm", "word", {"value": "köszönöm"}, "köszönöm"),
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
    # A bare "egy" is the indefinite article and a bare "első" means "first".
    ("egy alma", "egy alma"),
    ("az első", "az első"),
    ("két alma", "2 alma"),
    ("egyezer", "1000"),
    ("egymillió kétezer-egy", "1 002 001"),
    ("kukac a", "kukac a"),
    ("a b c", "a b c"),
    ("öt ( hat ) .", "5 (6)."),
    ("öt , hat", "5, 6"),
    ("két liter", "2l"),
    ("egy egész öt század dollár", "$1,05"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["kettő harmad", "nyolc óra öt perc", "egy dollár egy cent", "tízezer",
                                    "április elseje", "a b c kukac d e f pont hu"])
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
    spoken = 'út\\a"b'
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
