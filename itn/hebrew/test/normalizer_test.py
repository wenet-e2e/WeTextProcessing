# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

from itn.hebrew.inverse_normalizer import InverseNormalizer

PACKAGE = Path(__file__).resolve().parent.parent


def original_cases():
    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_*.txt")):
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


def test_original_fixture_count():
    assert len(CASES) == 337
    assert CASES[0].values == ("אפס", "אפס")
    # NeMo compares exactly; this measure input keeps its trailing space.
    assert CASES[235].values == ("מינוס שתים עשרה נקודה חמש מעלות ", "-12.5°")
    # Files without a trailing newline still count their last line.
    assert CASES[-1].values == ("יש לי חמישה תפוחים", "יש לי חמישה תפוחים")


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


@pytest.mark.parametrize("spoken, kind, members, written", [
    ("כמאה עשרים וחמש", "cardinal", {"prefix": "כ", "integer": "מאה עשרים וחמש"}, "כ-125"),
    ("מינוס שלוש", "cardinal", {"negative": "מינוס", "integer": "שלוש"}, "-3"),
    ("שלושה מיליוןאלף וחמש", "cardinal", {"integer": "שלושה מיליוןאלף וחמש"}, "3,001,005"),
    ("שבעים ותשע וחצי", "decimal", {"integer_part": "שבעים ותשע", "fraction": "וחצי"}, "79.5"),
    ("ארבע נקודה שמונים וחמש מיליארד", "decimal",
     {"integer_part": "ארבע", "fractional_part": "שמונים וחמש", "quantity": "מיליארד"}, "4.85 מיליארד"),
    ("עשר מיליון", "decimal", {"integer_part": "עשר", "quantity": "מיליון"}, "10 מיליון"),
    ("בשתיים וחצי מיליון", "decimal",
     {"prefix": "ב", "integer_part": "שתיים", "fraction": "וחצי", "quantity": "מיליון"}, "ב-2.5 מיליון"),
    ("סנטימטר אחד", "measure", {"units": "סנטימטר", "integer": "אחד"}, "1 ס״מ"),
    ("במאה אחוז", "measure", {"prefix": "ב", "integer": "מאה", "units": "אחוז"}, "ב-100%"),
    ("שלוש וחצי קילוגרם", "measure", {"integer_part": "שלוש", "fraction": "וחצי", "units": "קילוגרם"}, "3.5 ק״ג"),
    ("השתיים עשרה לשתיים עשרה אלף תשע מאות תשעים ואחת", "date",
     {"prefix": "ה", "day": "שתיים עשרה", "month": "לשתיים עשרה", "year": "אלף תשע מאות תשעים ואחת"}, "ה-12.12.1991"),
    ("בשבעה באוקטובר", "date", {"prefix": "ב", "day": "שבעה", "month": "באוקטובר"}, "ב-7.10"),
    ("בינואר עשרים עשרים ואחת", "date", {"month": "בינואר", "year": "עשרים עשרים ואחת"}, "בינואר 2021"),
    ("בשנת אלפיים וחמש", "date", {"year_prefix": "בשנת", "year": "אלפיים וחמש"}, "בשנת 2005"),
    ("רבע לשש בערב", "time", {"minutes": "רבע", "hours": "לשש", "suffix": "בערב"}, "17:45 בערב"),
    ("באחת ושתי דקות בצהריים", "time", {"prefix": "ב", "hours": "אחת", "minutes": "ושתי דקות", "suffix": "בצהריים"},
     "ב-13:02 בצהריים"),
    ("חמש לפנות ערב", "time", {"hours": "חמש", "suffix": "לפנות ערב"}, "17:00 לפנות ערב"),
    ("חצות ושתי דקות", "time", {"hours": "חצות", "minutes": "ושתי דקות"}, "0:02"),
    ("ארצות הברית", "whitelist", {"value": "ארצות הברית"}, "ארה״ב"),
    ("שלום", "word", {"value": "שלום"}, "שלום"),
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
    # The one-letter prefix stays on whitelisted abbreviations.
    ("לארצות הברית", "לארה״ב"),
    # A leading ו is the conjunction, kept as a prefix.
    ("ושלושים", "ו-30"),
    ("שלושים,", "30 ,"),
    # Bare digits stay words; negated or quantified they convert.
    ("חמש", "חמש"),
    ("מינוס אפס", "-0"),
    ("שני מיליון", "2 מיליון"),
    # Times convert only when the suffix fits the hour.
    ("שמונה בצהריים", "שמונה בצהריים"),
    ("שמונה בערב", "20:00 בערב"),
    ("שתיים לחמש", "4:58"),
    ("עשר בלילה", "22:00 בלילה"),
    ("אחת עשרה ועשרים בלילה", "23:20 בלילה"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["שתיים וחצי", "עשרים ושש באפריל", "רבע לשש", "מאה אלף", "אחד"])
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
    spoken = 'ע"פ\\x'
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
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.hebrew\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
