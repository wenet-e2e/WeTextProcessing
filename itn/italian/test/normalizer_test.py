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

from itn.italian.inverse_normalizer import InverseNormalizer

PACKAGE = Path(__file__).resolve().parent.parent
# NeMo has no Italian ITN fixtures; the TN fixtures are read in reverse.
TN_DATA = PACKAGE.parent.parent / "tn" / "italian" / "test" / "data"

# Reversed TN pairs that cannot hold, keyed by spoken form.
EXCLUDED = {
    # ``d`` for days is not written back: ``sette giorni`` (also a TN fixture) stays in words.
    "cento cinquantacinque giorni": "155 d",
    # Spacing only: ITN writes every currency symbol after a space (``3,23 €``).
    "tre euro ventitre centesimi": "3,23€",
    # TN reads ``4,2`` and ``4,20`` alike; ITN writes two-digit minor units (``4,20 £``).
    "quattro sterline venti penny": "4,2 £",
    # TN reads ``eur`` and ``€`` alike; ITN writes the symbol (``1 €``).
    "un euro": "1 eur",
    "un euro per il caffè": "1 eur per il caffè",
}


def reversed_cases():
    """Groups TN fixtures by spoken form; any written original is a correct answer."""

    groups = {}
    for path in sorted(TN_DATA.glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.endswith("\n"):
                    line = line[:-1]
                written, *spoken_forms = line.split("~")
                for spoken in spoken_forms:
                    groups.setdefault(spoken, []).append(written)
    return groups


GROUPS = reversed_cases()
CASES = [pytest.param(spoken, tuple(written), id=spoken) for spoken, written in GROUPS.items() if spoken not in EXCLUDED]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def test_reversed_fixture_count():
    # 52 TN lines, three of which list two spoken readings.
    assert sum(len(written) for written in GROUPS.values()) == 55
    assert len(GROUPS) == 55
    assert len(CASES) == 50
    for spoken, written in EXCLUDED.items():
        assert GROUPS[spoken] == [written]


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


@pytest.mark.parametrize("spoken, written", list(EXCLUDED.items()))
def test_excluded_pairs_have_a_canonical_reading(normalizer, spoken, written):
    assert normalizer.normalize(spoken) != written


def test_tag_then_verbalize_matches_normalize(normalizer):
    for spoken in list(GROUPS) + [case[0] for case in HAND_WRITTEN]:
        assert normalizer.verbalize(normalizer.tag(spoken)) == normalizer.normalize(spoken)


HAND_WRITTEN = [
    # cardinal
    ("centocinquantacinque", "155"),
    ("duemilaventisei", "2026"),
    ("mille novecento ottantaquattro", "1984"),
    ("ventitré", "23"),
    ("centottanta", "180"),
    ("ventunomila", "21000"),
    ("tre miliardi quattrocento cinquantatre milioni", "3453000000"),
    ("meno quarantotto", "-48"),
    ("zero", "0"),
    ("sette giorni", "7 giorni"),
    # decimal
    ("zero virgola zero cinque", "0,05"),
    ("meno tre virgola cinque", "-3,5"),
    ("dodici virgola cento uno", "12,101"),
    ("due milioni", "2 milioni"),
    ("un miliardo di euro", "1 miliardo di euro"),
    ("sei virgola dodici trentaquattro", "6,1234"),
    # measure
    ("venti gradi celsius", "20 °C"),
    ("cento chilometri per ora", "100 km/h"),
    ("tre chilogrammi", "3 kg"),
    ("cinque per cento", "5%"),
    ("due metri quadrati", "2 m²"),
    ("una tonnellata", "1 t"),
    ("due virgola cinque litri", "2,5 l"),
    # money
    ("cinquanta centesimi", "0,50 €"),
    ("tre euro e venti centesimi", "3,20 €"),
    ("due virgola cinque euro", "2,5 €"),
    ("dieci dollari", "10 $"),
    ("una sterlina", "1 £"),
    ("dieci franchi svizzeri", "10 CHF"),
    ("venti rupie e cinquanta paise", "20,50 ₹"),
    ("un dollaro e un centesimo", "1,01 $"),
    # time
    ("cinque e un quarto", "05:15"),
    ("cinque meno un quarto", "04:45"),
    ("nove meno dieci minuti", "08:50"),
    ("alle dodici e trenta", "alle 12:30"),
    ("dalle nove alle diciotto", "dalle 09:00 alle 18:00"),
    ("a mezzogiorno", "a 12:00"),
    ("tre e trenta minuti e un secondo", "03:30:01"),
    ("otto e mezza", "08:30"),
    # electronic
    ("m a r i o punto r o s s i chiocciola gmail punto com", "mario.rossi@gmail.com"),
    ("w w w punto g o o g l e punto IT", "www.google.it"),
    ("h t t p due punti slash slash e x a m p l e punto org", "http://example.org"),
    ("chiocciola m a r i o", "@mario"),
    ("s i t o punto i t", "sito.it"),
    ("i n f o trattino basso uno chiocciola s i t o punto net", "info_1@sito.net"),
    ("a chiocciola b punto com e poi", "a@b.com e poi"),
    ("w w w punto nvidia punto com e poi", "www.nvidia.com e poi"),
    # whitelist
    ("Dottor Rossi", "Dott. Rossi"),
    ("professoressa Bianchi", "prof.ssa Bianchi"),
    ("eccetera", "ecc."),
    ("come volevasi dimostrare", "c.v.d."),
    ("viale Mazzini", "v.le Mazzini"),
    # word
    ("un uomo", "un uomo"),
    ("una settimana", "una settimana"),
    ("il punto e la virgola", "il punto e la virgola"),
    ("una chiocciola", "una chiocciola"),
    ("sette e trenta", "7 e 30"),
    ("perché", "perché"),
    # punctuation
    ("( cinque ) .", "(5)."),
    ("ciao , come stai ?", "ciao, come stai?"),
    ("{ uno }", "{1}"),
    ("sei - sette", "6 - 7"),
    ("evviva !", "evviva!"),
]


@pytest.mark.parametrize("spoken, written", HAND_WRITTEN)
def test_hand_written(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, members, written", [
    ("a chiocciola b punto com e poi", {"username": "a", "domain": "b punto com"}, "a@b.com e poi"),
    ("w w w punto nvidia punto com e poi", {"protocol": "w w w punto", "domain": "nvidia punto com"},
     "www.nvidia.com e poi"),
])
def test_electronic_boundaries_and_mapping(normalizer, spoken, members, written):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert parser.tokens[0].name == "electronic"
    assert parser.tokens[0].members == members
    assert normalizer.verbalize(tagged) == written
    results = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == normalizer.normalize(spoken, nbest=3)
    assert results[0].output_text == written
    for result in results:
        for mapping in result.mappings:
            assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken, kind, members, written", [
    ("cento cinquantacinque", "cardinal", {"integer": "cento cinquantacinque"}, "155"),
    ("meno cinque", "cardinal", {"negative": "meno", "integer": "cinque"}, "-5"),
    ("quattro virgola cinquantasei milioni", "decimal",
     {"integer_part": "quattro", "fractional_part": "cinquantasei", "quantity": "milioni"}, "4,56 milioni"),
    ("cinque chilometri per secondo", "measure", {"integer": "cinque", "units": "chilometri per secondo"}, "5 km/s"),
    ("quattro virgola tre litri", "measure", {"integer_part": "quattro", "fractional_part": "tre", "units": "litri"},
     "4,3 l"),
    ("due rubli un copeca", "money",
     {"integer_part": "due", "currency": "rubli", "fractional_part": "un", "minor": "copeca"}, "2,01 ₽"),
    ("cinquanta centesimi", "money", {"fractional_part": "cinquanta", "minor": "centesimi"}, "0,50 €"),
    ("diciassette e quindici minuti e ventisei secondi", "time",
     {"hours": "diciassette", "minutes": "quindici minuti", "seconds": "ventisei secondi"}, "17:15:26"),
    ("cinque meno un quarto", "time", {"hours": "cinque", "relation": "meno", "minutes": "un quarto"}, "04:45"),
    ("alle sedici", "time", {"preposition": "alle", "hours": "sedici"}, "alle 16:00"),
    ("h t t p s due punti slash slash w w w punto nvidia punto com", "electronic",
     {"protocol": "h t t p s due punti slash slash w w w punto", "domain": "nvidia punto com"},
     "https://www.nvidia.com"),
    ("chiocciola j e n s e n", "electronic", {"username": "j e n s e n"}, "@jensen"),
    ("signorina", "whitelist", {"value": "signorina"}, "sig.na"),
    ("caffè", "word", {"value": "caffè"}, "caffè"),
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


@pytest.mark.parametrize("spoken", ["dodici e mezza", "due rubli un copeca", "( cinque ) .", "signorina Sonia"])
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
    spoken = 'perché\\a"b'
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
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.italian\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
