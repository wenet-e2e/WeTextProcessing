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

from itn.armenian.inverse_normalizer import InverseNormalizer

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
    assert len(CASES) == 160
    # The whitelist fixture has no trailing newline; its last line still counts.
    assert CASES[109].values == ("քրիստոսից ծնունդից հետո", "Ք.ծ.հ.")
    assert CASES[110].values == ("", "")


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
    ("իննսունյոթ", "cardinal", {"integer": "իննսունյոթ"}, "97"),
    ("հինգ ու կես միլիոն", "decimal", {"integer_part": "հինգ", "fractional_part": "կես", "quantity": "միլիոն"},
     "5.5 միլիոն"),
    ("ութ հարյուր տրիլիոն", "decimal", {"integer_part": "ութ հարյուր", "quantity": "տրիլիոն"}, "800 տրիլիոն"),
    ("երկու երրորդ", "fraction", {"numerator": "երկու", "denominator": "երրորդ"}, "2/3"),
    ("հիսունյոթերորդ", "ordinal", {"integer": "հիսունյոթերորդ"}, "57-րդ"),
    ("հինգին տասնհինգ պակաս", "time", {"hours": "հինգին", "minutes": "տասնհինգ", "relation": "պակաս"}, "04:45"),
    ("տասներեք անց կես", "time", {"hours": "տասներեք", "relation": "անց", "minutes": "կես"}, "13:30"),
    ("իննսունյոթ հերց", "measure", {"integer": "իննսունյոթ", "units": "հերց"}, "97 Հց"),
    ("հինգից տասը դար", "measure", {"from": "հինգից", "to": "տասը", "units": "դար"}, "5-10 դ."),
    ("քսան հազար դրամ", "money", {"integer_part": "քսան հազար", "currency": "դրամ"}, "20000 ֏"),
    ("մեր թվարկությունից առաջ", "whitelist", {"value": "մեր թվարկությունից առաջ"}, "մ.թ.ա."),
    ("բարև", "word", {"value": "բարև"}, "բարև"),
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
    ("քսան։", "20 ։"),
    ("մեկ հարյուր", "1 100"),
    ("զրո զրո", "0 0"),
    ("ամբողջ հինգ", ".5"),
    ("մեկին հինգ պակաս", "00:55"),
    ("երկու հազար քսաներեք թվական", "2023 թ."),
    ("տասներկու դար", "12 դ."),
    ("ութ անց", "8 անց"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["մեկ երկրորդ", "հինգ անց կես", "հինգ յեն", "առաջին", "սույն թվականի"])
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
    spoken = 'ուղի\\a"b'
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
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.armenian\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
