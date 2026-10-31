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

from itn.marathi.inverse_normalizer import InverseNormalizer

PACKAGE = Path(__file__).resolve().parent.parent


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


def test_original_fixture_count():
    # No fixture file ends with a newline; every last line still counts.
    assert len(CASES) == 131
    assert CASES[32].values == ("उणे तेहतीस कोटी तेहतीस लाख", "-३३३३०००००")
    assert CASES[120].values == ("", "")
    assert CASES[122].values == ("म ", "म")
    assert CASES[130].values == ("पिणे", "पिणे")


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
    ("उणे सातशे बावीस", "cardinal", {"negative": "उणे", "integer": "सातशे बावीस"}, "-७२२"),
    ("दोन लाख एकतीसशे", "cardinal", {"integer": "दोन लाख एकतीसशे"}, "२०३१००"),
    ("एक अब्ज दोन", "cardinal", {"integer": "एक अब्ज दोन"}, "१" + "०" * 8 + "२"),
    ("उणे बारा पूर्णांक एक पाच कोटी", "decimal", {
        "negative": "उणे", "integer_part": "बारा", "fractional_part": "एक पाच", "quantity": "कोटी",
    }, "-१२.१५ कोटी"),
    ("सतरा लाख", "decimal", {"integer_part": "सतरा", "quantity": "लाख"}, "१७ लाख"),
    ("पूर्णांक पाच", "decimal", {"fractional_part": "पाच"}, ".५"),
    ("पंधरा फेब्रुवारी अठराशे सतरा", "date", {"day": "पंधरा", "month": "फेब्रुवारी", "year": "अठराशे सतरा"},
     "१५ फेब्रुवारी १८१७"),
    ("जानेवारी दहा", "date", {"month": "जानेवारी", "day": "दहा"}, "जानेवारी १०"),
    ("जानेवारी दोन हजार", "date", {"month": "जानेवारी", "year": "दोन हजार"}, "जानेवारी २०००"),
    ("इसवी सन पूर्व चारशे", "date", {"era": "इसवी सन पूर्व", "year": "चारशे"}, "इ.स.पू. ४००"),
    ("अकरा वाजून दोन मिनिटे", "time", {"hours": "अकरा", "minutes": "दोन"}, "११:०२"),
    ("अकराला पाच मिनिटे", "time", {"hours": "अकराला", "minutes": "पाच"}, "१०:५५"),
    ("पावणे एक", "time", {"relation": "पावणे", "hours": "एक"}, "१२:४५"),
    ("साडे चार", "time", {"relation": "साडे", "hours": "चार"}, "०४:३०"),
    ("अडीच", "time", {"hours": "अडीच"}, "०२:३०"),
    ("आई", "word", {"value": "आई"}, "आई"),
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
    ("वीस।", "२० ।"),
    ("नव्व्याण्णव अब्ज नव्व्याण्णव कोटी नव्व्याण्णव लाख नव्व्याण्णव हजार नऊशे नव्व्याण्णव", "९९९९९९९९९९९"),
    ("शून्य शून्य", "० ०"),
    ("तीन पूर्णांक पाच हजार", "३.५ हजार"),
    ("बारा वाजून एक मिनिट", "१२:०१"),
    ("सव्वा बारा वाजता", "१२:१५ वाजता"),
    ("मार्च पाच", "मार्च ५"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["चार लाख", "साडे चार", "जानेवारी दहा", "तीन हजार एकशे"])
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
    spoken = 'मी\\a"b'
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
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.marathi\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
