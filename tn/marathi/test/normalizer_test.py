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

from tn.marathi.normalizer import Normalizer

PACKAGE = Path(__file__).resolve().parent.parent
# NeMo has no Marathi TN fixtures; the ITN fixtures are read in reverse.
ITN_DATA = PACKAGE.parent.parent / "itn" / "marathi" / "test" / "data"

# Reversed ITN pairs that cannot hold for a deterministic TN, with reasons.
EXCLUDED = {
    "५८०१": "ITN accepts the colloquial hundreds count; TN reads cardinals with हजार",
    "७५१६": "ITN accepts the colloquial hundreds count; TN reads cardinals with हजार",
    "-५२००": "ITN accepts the colloquial hundreds count; TN reads cardinals with हजार",
    "१०:५८": "fixture uses singular मिनिट for two minutes, unlike आठला सात मिनिटे",
    "०६:४८": "fixture reads :48 as minutes-to, but ०२:४८ as minutes-past",
    "०९:३०": "fixture reads :30 as वाजून तीस मिनिटे, but ०४:३० as साडे",
    "म": "spoken side keeps an untrimmed trailing space",
}


def reversed_cases():
    candidates = {}
    for path in sorted(ITN_DATA.glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for line in stream:
                if line.endswith("\n"):
                    line = line[:-1]
                spoken, written = line.split("~")
                candidates.setdefault(written, []).append(spoken)
    return [pytest.param(written, spoken, id=written) for written, spoken in candidates.items()]


CASES = reversed_cases()
CHECKED = [param for param in CASES if param.values[0] not in EXCLUDED]


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def test_reversed_fixture_count():
    assert sum(len(param.values[1]) for param in CASES) == 131
    assert len(CASES) == 128
    assert len(CHECKED) == 121
    assert set(EXCLUDED) <= {param.values[0] for param in CASES}


@pytest.mark.parametrize("written, spoken", CHECKED)
def test_reversed_original(normalizer, written, spoken):
    assert normalizer.normalize(written) in spoken


@pytest.mark.parametrize("written, spoken", CHECKED)
def test_reversed_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text in spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        written = param.values[0]
        assert normalizer.verbalize(normalizer.tag(written)) == normalizer.normalize(written)


@pytest.mark.parametrize("written, spoken", [
    ("५८०१", "पाच हजार आठशे एक"),
    ("१०:५८", "अकराला दोन मिनिटे"),
    ("०६:४८", "सहा वाजून अठ्ठेचाळीस मिनिटे"),
    ("०९:३०", "साडे नऊ"),
    ("म", "म"),
])
def test_excluded_pairs_read_consistently(normalizer, written, spoken):
    assert written in EXCLUDED
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written, kind, members, spoken", [
    ("-722", "cardinal", {"negative": "-", "integer": "722"}, "उणे सातशे बावीस"),
    ("१,२३,४५६", "cardinal", {"integer": "१,२३,४५६"}, "एक लाख तेवीस हजार चारशे छप्पन"),
    ("007", "cardinal", {"integer": "007"}, "शून्य शून्य सात"),
    ("-१२.१५ कोटी", "decimal", {
        "negative": "-", "integer_part": "१२", "fractional_part": "१५", "quantity": "कोटी",
    }, "उणे बारा पूर्णांक एक पाच कोटी"),
    ("4 लाख", "decimal", {"integer_part": "4", "quantity": "लाख"}, "चार लाख"),
    ("15 फेब्रुवारी 1817", "date", {"day": "15", "month": "फेब्रुवारी", "year": "1817"},
     "पंधरा फेब्रुवारी अठराशे सतरा"),
    ("मार्च ०५", "date", {"month": "मार्च", "day": "०५"}, "मार्च पाच"),
    ("इ.स.पू. ४००", "date", {"era": "इ.स.पू.", "year": "४००"}, "इसवी सन पूर्व चारशे"),
    ("11:02", "time", {"hours": "11", "minutes": "02"}, "अकरा वाजून दोन मिनिटे"),
    ("१३:३०", "time", {"hours": "१३", "minutes": "३०"}, "दीड"),
    ("२३:५९", "time", {"hours": "२३", "minutes": "५९"}, "बाराला एक मिनिट"),
    ("१२:००", "time", {"hours": "१२", "minutes": "००"}, "बारा वाजता"),
    ("आई", "word", {"value": "आई"}, "आई"),
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
    ("(५)", "( पाच )"),
    ("५।", "पाच ।"),
    ("१००", "शंभर"),
    ("११००", "एक हजार एकशे"),
    ("1,000,000", "दहा लाख"),
    ("९९९९९९९९९९९", "नव्व्याण्णव अब्ज नव्व्याण्णव कोटी नव्व्याण्णव लाख नव्व्याण्णव हजार नऊशे नव्व्याण्णव"),
    ("१२३४५६७८९०१२", "एक दोन तीन चार पाच सहा सात आठ नऊ शून्य एक दोन"),
    (".५", "पूर्णांक पाच"),
    ("३.५ हजार", "तीन पूर्णांक पाच हजार"),
    ("२२ मे 2023", "बावीस मे दोन हजार तेवीस"),
    ("३२ जून", "बत्तीस जून"),
    ("00:55", "एकला पाच मिनिटे"),
    ("7:15", "सव्वा सात"),
    ("०२:३०", "अडीच"),
    ("१४:४५", "पावणे तीन"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


def test_quarter_times_keep_plain_reading_as_alternative(normalizer):
    assert normalizer.normalize("०४:३०", nbest=2) == ["साडे चार", "चार वाजून तीस मिनिटे"]


@pytest.mark.parametrize("written", ["४ लाख", "०४:३०", "मार्च ५ १९९८", "१,००,०००", "-१२.५"])
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
    written = 'मी\\a"b'
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
    itn_import = re.compile(r"^\s*(from|import)\s+itn\.marathi\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not itn_import.search(path.read_text(encoding="utf-8")), path
