# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

from itn.vietnamese.inverse_normalizer import InverseNormalizer


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


def test_original_fixture_count_and_whitespace():
    assert len(CASES) == 453
    assert ("  ba   mươi   mốt  ", "31") in [case.values for case in CASES]
    assert ("", "") in [case.values for case in CASES]


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
        if spoken:
            assert normalizer.verbalize(normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken, kinds, members", [
    ("âm hai mươi lăm", ["cardinal"], [{"negative": "âm", "integer": "hai mươi lăm"}]),
    ("  ba   mươi   mốt  ", ["cardinal"], [{"integer": "ba   mươi   mốt"}]),
    ("một triệu hai", ["decimal"], [{"integer_part": "một", "quantity": "triệu", "fractional_part": "hai"}]),
    ("trừ năm phần mười hai", ["fraction"], [{"negative": "trừ", "numerator": "năm", "denominator": "mười hai"}]),
    ("thứ nhất", ["ordinal"], [{"integer": "nhất"}]),
    ("ngày mười tháng năm", ["word", "date"], [{"value": "ngày"}, {"day": "mười", "month": "năm"}]),
    ("mười sáu giờ kém mười lăm", ["time"], [{"hours": "mười sáu", "relation": "kém", "minutes": "mười lăm"}]),
    ("hai đô rưỡi", ["money"], [{"integer_part": "hai", "currency": "đô", "minor_part": "rưỡi"}]),
    ("hai mét rưỡi", ["measure"], [{"integer": "hai", "units": "mét", "fractional_part": "rưỡi"}]),
    ("cộng tám mươi bốn không chín ba sáu năm năm năm bốn bốn chín", ["telephone"],
     [{"country_code": "cộng tám mươi bốn", "number_part": "không chín ba sáu năm năm năm bốn bốn chín"}]),
    ("a b c a móc g mail chấm com", ["electronic"], [{"username": "a b c", "domain": "g mail chấm com"}]),
])
def test_raw_fields(normalizer, spoken, kinds, members):
    tagged = normalizer.tag(spoken)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == kinds
    assert [token.members for token in parser.tokens] == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken, written", [
    ("một triệu đô năm mươi", "1.50 triệu$"),
    ("hai đô năm mươi", "2.50$"),
    ("ba giờ  mười giây", "3:00:10"),
    ("một trăm,", "100 ,"),
    ("không không không", "000"),
])
def test_additional_semantics(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken", ["một trăm linh hai", "hai đô rưỡi", "mười sáu giờ kém mười lăm", "năm năm"])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)


def test_escaped_word_round_trips(normalizer):
    spoken = 'đường\\dẫn"{x} hai đô'
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == 'đường\\dẫn"{x} 2$'
    assert [(m.input_text, m.output_text, m.token_type) for m in result.mappings] == [
        ('đường\\dẫn"{x}', 'đường\\dẫn"{x}', "word"),
        ("hai đô", "2$", "money"),
    ]
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert parser.tokens[0].members == {"value": 'đường\\dẫn"{x}'}


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
