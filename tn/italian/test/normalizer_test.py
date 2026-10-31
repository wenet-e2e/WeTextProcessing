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

from tn.italian.normalizer import Normalizer

PACKAGE = Path(__file__).resolve().parent.parent


def original_cases():
    """Reads ``written~spoken[~spoken...]`` lines; any listed reading is correct."""

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
    assert len(CASES) == 52
    # Files without a trailing newline still count their last line.
    assert CASES[12].values == ("7 giorni sono una settimana", ("sette giorni sono una settimana",))
    # Time fixtures list alternative readings.
    assert CASES[42].values == ("12:30", ("dodici e trenta minuti", "dodici e mezza"))


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
        written, _ = param.values
        assert normalizer.verbalize(normalizer.tag(written)) == normalizer.normalize(written)


@pytest.mark.parametrize("written, kind, members, spoken", [
    ("155", "cardinal", {"integer": "155"}, "cento cinquantacinque"),
    ("1.000.000", "cardinal", {"integer": "1.000.000"}, "un milione"),
    ("-21", "cardinal", {"negative": "-", "integer": "21"}, "meno ventuno"),
    ("5,342", "decimal", {"integer_part": "5", "fractional_part": "342"}, "cinque virgola trecento quarantadue"),
    ("3,55 miliardi", "decimal", {"integer_part": "3", "fractional_part": "55", "quantity": "miliardi"},
     "tre virgola cinquantacinque miliardi"),
    ("5 km/s", "measure", {"integer": "5", "units": "km/s"}, "cinque chilometri per secondo"),
    ("1 m", "measure", {"integer": "1", "units": "m"}, "un metro"),
    ("4,3 l", "measure", {"integer_part": "4", "fractional_part": "3", "units": "l"}, "quattro virgola tre litri"),
    ("18%", "measure", {"integer": "18", "units": "%"}, "diciotto percento"),
    ("2,01 ₽", "money", {"integer_part": "2", "fractional_part": "01", "currency": "₽"}, "due rubli un copeca"),
    ("€1", "money", {"currency": "€", "integer_part": "1"}, "un euro"),
    ("1 £", "money", {"integer_part": "1", "currency": "£"}, "una sterlina"),
    ("0,50 $", "money", {"integer_part": "0", "fractional_part": "50", "currency": "$"}, "cinquanta centesimi"),
    ("2 milioni €", "money", {"integer_part": "2", "quantity": "milioni", "currency": "€"},
     "due milioni di euro"),
    ("17:15:26", "time", {"hours": "17", "minutes": "15", "seconds": "26"},
     "diciassette e quindici minuti e ventisei secondi"),
    ("16:00", "time", {"hours": "16", "minutes": "00"}, "sedici"),
    ("05:01", "time", {"hours": "05", "minutes": "01"}, "cinque e un minuto"),
    ("https://www.nvidia.com", "electronic", {"protocol": "https://www.", "domain": "nvidia.com"},
     "h t t p s due punti slash slash w w w punto nvidia punto com"),
    ("abc.def@23df.sd.IT", "electronic", {"username": "abc.def", "domain": "23df.sd.IT"},
     "a b c punto d e f chiocciola due tre d f punto s d punto I T"),
    ("@jensen", "electronic", {"username": "jensen"}, "chiocciola j e n s e n"),
    ("mail.nasa.gov", "electronic", {"domain": "mail.nasa.gov"}, "m a i l punto n a s a punto gov"),
    ("sig.na", "whitelist", {"value": "sig.na"}, "signorina"),
    ("caffè", "word", {"value": "caffè"}, "caffè"),
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
    ("(5)", "(cinque)"),
    ("21000", "ventunomila"),
    ("1001", "mille uno"),
    ("1000000000000", "un miliardo"),
    ("3,00 €", "tre euro"),
    ("£4,20", "quattro sterline venti penny"),
    ("2,5 kg", "due virgola cinque chilogrammi"),
    ("10 cm", "dieci centimetri"),
    ("1.000", "mille"),
    ("Dott. Rossi", "dottor Rossi"),
    ("mario.rossi@gmail.com", "m a r i o punto r o s s i chiocciola gmail punto com"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written, spoken, fields", [
    ("(5)", "(cinque)", [("punctuation", {"value": "("}), ("cardinal", {"integer": "5"}),
                         ("punctuation", {"value": ")"})]),
    ("5,", "cinque,", [("cardinal", {"integer": "5"}), ("punctuation", {"value": ","})]),
    ("5 !", "cinque !", [("cardinal", {"integer": "5"}), ("punctuation", {"value": " !"})]),
    ("( 5 )", "( cinque )", [("punctuation", {"value": "( "}), ("cardinal", {"integer": "5"}),
                             ("punctuation", {"value": " )"})]),
])
def test_punctuation_spacing(normalizer, written, spoken, fields):
    assert normalizer.normalize(written) == spoken
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [(token.name, token.members) for token in parser.tokens] == fields
    assert normalizer.verbalize(tagged) == spoken
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert outputs[0] == spoken
    assert len(outputs) == len(set(outputs))
    assert [result.output_text for result in mapped] == outputs
    for result in mapped:
        for mapping in result.mappings:
            assert written[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_idiomatic_times_are_alternatives(normalizer):
    assert normalizer.normalize("12:30", nbest=2) == ["dodici e trenta minuti", "dodici e mezza"]
    assert normalizer.normalize("05:15", nbest=2) == ["cinque e quindici minuti", "cinque e un quarto"]


@pytest.mark.parametrize("written", ["12:30", "2,01 ₽", "5 km/s", "https://www.nvidia.com", "sig.na"])
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
    written = 'perché\\a"b'
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
    itn_import = re.compile(r"^\s*(from|import)\s+itn\.italian\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not itn_import.search(path.read_text(encoding="utf-8")), path
