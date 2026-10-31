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
from collections import defaultdict
from pathlib import Path

import pytest

from tn.hebrew.normalizer import Normalizer

PACKAGE = Path(__file__).resolve().parent.parent
ITN_DATA = PACKAGE.parent.parent / "itn" / "hebrew" / "test" / "data"

# Reversed ITN pairs whose spoken side TN cannot produce. The fixtures read
# the same construction in different ways, so one canonical reading must
# miss some of them.
EXCLUDED = {
    "77": "masculine units (שבעים ושבעה); counting is feminine elsewhere",
    "330": "ו before round tens after hundreds; 230, 990 and 1,130 omit it",
    "-3,220": "ו before round tens after hundreds; 230, 990 and 1,130 omit it",
    "450 מ״ג": "ו before round tens after hundreds; 230, 990 and 1,130 omit it",
    "כ-101,200": "ו before hundreds after thousands; 2,100 and 10,200 omit it",
    "1,000,015": "no ו before a final teen; 2,011 and 10,015 use it",
    "10,201": "masculine final one (ואחד); counting is feminine elsewhere",
    "-101,220": "feminine one before אלף; 101,000 and 101,200 use ואחד אלף",
    "-101,229": "feminine one before אלף; 101,000 and 101,200 use ואחד אלף",
    "3,001,005": "the only spoken form glues מיליוןאלף without a space",
    "אנחנו היינו איזה 15 איש": "masculine teen (חמישה עשר); counting is feminine elsewhere",
    "1,000%": "plural אחוזים; every other percentage uses אחוז",
    "79.5": "fraction word for an integer above 9; 12.5 and 18.25 use נקודה",
    "10.2": "fraction word for .2; 5.2 and 60.2 use נקודה",
    "1.25 מיליארד": "נקודה for .25 of 1; 3.25-style amounts use ורבע (5.25, 8.25)",
    "1.59 מיליארד": "digit-by-digit .59; 1.84 and 18.56 read two digits as a number",
    "18.25": "digit-by-digit .25; 18.85 and 18.22 read two digits as a number",
    "-12.5°": "spelling שתים and a trailing space in the spoken form",
    "ה-12.12.1991": "feminine day (השתיים עשרה); days are masculine (שבעה עשר באוגוסט)",
    "26.4.2024": "feminine or ordinal day; days are masculine (26.4.24 accepts עשרים ושישה)",
    "ב-3.3.1990": "ordinal day and ordinal month (בשלישי לשלישי)",
    "היה בערך בעשירי ב-10.5": "ordinal day after ב and month with ל (בעשירי למאי)",
    "1.5": "day.month without year or prefix is a decimal (1.1, 18.6)",
    "18.1": "day.month without year or prefix is a decimal (1.1, 18.6)",
    "אני מדבר על 18.4": "day.month without year or prefix is a decimal (1.1, 18.6)",
    "אמרתי לו שיש לי כאבים ביד, אז הוא בדק אותי": "the spoken form ends with a space",
    "בכל כיתה יש 20 , 22 תלמידים": "the written comma is a separate token; the spoken one is attached",
    "מחר ב-6:30 בבוקר נעלה על האוטובוסים ונצא לטיול השנתי. בשעה שמונה נגיע למצדה ונתחיל לטפס למעלה, נהיה שם עד "
    "13:30 בצהריים , אולי 1:45 , ונרד בשביל הנחש": "the written commas are separate tokens; the spoken ones are attached",
}


def reversed_cases():
    candidates = defaultdict(set)
    for path in sorted(ITN_DATA.glob("test_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.endswith("\n"):
                    line = line[:-1]
                spoken, written = line.split("~")
                candidates[written].add(spoken)
    return candidates


CANDIDATES = reversed_cases()
CASES = [pytest.param(written, spoken, id=written[:40]) for written, spoken in CANDIDATES.items()
         if written not in EXCLUDED]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def test_reversed_fixture_count():
    # 337 ITN lines hold 330 distinct pairs.
    assert sum(len(spoken) for spoken in CANDIDATES.values()) == 330
    assert len(CANDIDATES) == 296
    assert set(EXCLUDED) <= set(CANDIDATES)
    assert len(CASES) == 296 - len(EXCLUDED) == 268


@pytest.mark.parametrize("written, spoken", CASES)
def test_reversed(normalizer, written, spoken):
    assert normalizer.normalize(written) in spoken


@pytest.mark.parametrize("written", sorted(EXCLUDED))
def test_exclusions_still_needed(normalizer, written):
    assert normalizer.normalize(written) not in CANDIDATES[written]


@pytest.mark.parametrize("written, spoken", CASES)
def test_reversed_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text in spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for written in CANDIDATES:
        assert normalizer.verbalize(normalizer.tag(written)) == normalizer.normalize(written)


@pytest.mark.parametrize("written, kind, members, spoken", [
    ("23", "cardinal", {"integer": "23"}, "עשרים ושלוש"),
    ("כ-1,000", "cardinal", {"prefix": "כ", "integer": "1,000"}, "כאלף"),
    ("-3,721", "cardinal", {"negative": "-", "integer": "3,721"}, "מינוס שלושת אלפים שבע מאות עשרים ואחת"),
    ("101000", "cardinal", {"integer": "101000"}, "מאה ואחד אלף"),
    ("3043061", "cardinal", {"integer": "3043061"}, "שלושה מיליון ארבעים ושלוש אלף שישים ואחת"),
    ("2.75", "decimal", {"integer_part": "2", "fractional_part": "75"}, "שתיים ושלושת רבעי"),
    ("18.05", "decimal", {"integer_part": "18", "fractional_part": "05"}, "שמונה עשרה נקודה אפס חמש"),
    ("1.84 מיליארד", "decimal", {"integer_part": "1", "fractional_part": "84", "quantity": "מיליארד"},
     "אחד נקודה שמונים וארבע מיליארד"),
    ("2 מיליון", "decimal", {"integer_part": "2", "quantity": "מיליון"}, "שני מיליון"),
    ("ב-2.5 מיליון", "decimal", {"prefix": "ב", "integer_part": "2", "fractional_part": "5", "quantity": "מיליון"},
     "בשתיים וחצי מיליון"),
    ("-15%", "measure", {"negative": "-", "integer": "15", "units": "%"}, "מינוס חמש עשרה אחוז"),
    ("1 ס״מ", "measure", {"integer": "1", "units": "ס״מ"}, "סנטימטר אחד"),
    ("ל-4 ס״מ", "measure", {"prefix": "ל", "integer": "4", "units": "ס״מ"}, "לארבע סנטימטר"),
    ("3.5 ק״ג", "measure", {"integer_part": "3", "fractional_part": "5", "units": "ק״ג"}, "שלוש וחצי קילוגרם"),
    ("12:35 בצהריים", "time", {"hours": "12", "minutes": "35", "suffix": "בצהריים"},
     "עשרים וחמישה לאחת בצהריים"),
    ("ב-18:20 בערב", "time", {"prefix": "ב", "hours": "18", "minutes": "20", "suffix": "בערב"},
     "בשש ועשרים דקות בערב"),
    ("0:12", "time", {"hours": "0", "minutes": "12"}, "חצות ושתיים עשרה דקות"),
    ("15:30", "time", {"hours": "15", "minutes": "30"}, "חמש עשרה וחצי"),
    ("ב-17.8.2030", "date", {"prefix": "ב", "day": "17", "month": "8", "year": "2030"},
     "בשבעה עשר באוגוסט עשרים שלושים"),
    ("ה-10.4", "date", {"prefix": "ה", "day": "10", "month": "4"}, "העשירי באפריל"),
    ("1.5.1983", "date", {"day": "1", "month": "5", "year": "1983"}, "אחד במאי אלף תשע מאות שמונים ושלוש"),
    ("ביוני 2015", "date", {"month": "ביוני", "year": "2015"}, "ביוני אלפיים וחמש עשרה"),
    ("בשנת 70", "date", {"year_prefix": "בשנת", "year": "70"}, "בשנת שבעים"),
    ("לארה״ב", "whitelist", {"value": "לארה״ב"}, "לארצות הברית"),
    ("שלום", "word", {"value": "שלום"}, "שלום"),
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
    ("(5)", "( חמש )"),
    ("20,", "עשרים ,"),
    ("0", "אפס"),
    ("1 מיליון", "מיליון"),
    (".5", "נקודה חמש"),
    ("23:59", "דקה לחצות"),
    ("05:58", "שתי דקות לשש"),
    ("17:00 בבוקר", "שבע עשרה בבוקר"),
    ("ב-30.6", "בשלושים ביוני"),
    ("5 ק\"ג", "חמש קילוגרם"),
    ("1,234,567,890", "1,234,567,890"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written", ["2.5", "17:45 בערב", "ה-10.4", "1 ס״מ", "ארה״ב"])
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
    written = 'ע"פ\\x'
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
    itn_import = re.compile(r"^\s*(from|import)\s+itn\.hebrew\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not itn_import.search(path.read_text(encoding="utf-8")), path
