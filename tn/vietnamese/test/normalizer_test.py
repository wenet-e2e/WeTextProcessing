# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.vietnamese.normalizer import Normalizer


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
    assert len(CASES) == 472
    assert CASES[0].values == ("1", "một")


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


@pytest.mark.parametrize("written, kinds, members", [
    ("23", ["cardinal"], [{"integer": "23"}]),
    ("-1.234", ["cardinal"], [{"negative": "-", "integer": "1.234"}]),
    ("-0,007", ["decimal"], [{"negative": "-", "integer_part": "0", "fractional_part": "007"}]),
    ("1tr2", ["decimal"], [{"integer_part": "1", "quantity": "tr", "minor_part": "2"}]),
    ("-12 1/4", ["fraction"], [{"negative": "-", "integer_part": "12", "numerator": "1", "denominator": "4"}]),
    ("thứ 4", ["ordinal"], [{"integer": "4"}]),
    ("2024/01/15", ["date"], [{"year": "2024", "month": "01", "day": "15"}]),
    ("năm 200 TCN", ["date"], [{"year": "200", "era": "TCN"}]),
    ("8:23 gmt", ["time"], [{"hours": "8", "minutes": "23", "zone": "gmt"}]),
    ("10,5$", ["money"], [{"integer_part": "10", "fractional_part": "5", "currency_maj": "$"}]),
    ("10,5 đồng", ["money"], [{"integer_part": "10", "fractional_part": "5", "currency": "đồng"}]),
    ("100₫/kg", ["money"], [{"integer_part": "100", "currency_maj": "₫", "per_unit": "/kg"}]),
    ("120km/h", ["measure"], [{"integer": "120", "units": "km/h"}]),
    ("thế kỷ XV", ["roman"], [{"key": "thế kỷ", "integer": "XV"}]),
    ("10:00-11:00", ["time", "range", "time"],
     [{"hours": "10", "minutes": "00"}, {"value": "-"}, {"hours": "11", "minutes": "00"}]),
])
def test_raw_fields(normalizer, written, kinds, members):
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    assert [token.name for token in parser.tokens] == kinds
    assert [token.members for token in parser.tokens] == members
    assert normalizer.verbalize(tagged) == normalizer.normalize(written)


def test_iso_date_is_reordered(normalizer):
    assert normalizer.normalize("2024/01/15") == "ngày mười lăm tháng một năm hai nghìn hai mươi tư"


@pytest.mark.parametrize("written, spoken", [
    ("(123)", "(một trăm hai mươi ba)"),
    ("a ,b", "a,b"),
    ("Hello, world!", "Hello, world!"),
    ("0p", "không phút"),
    ("00:00:05", "không giờ năm giây"),
    ("x 5$$", "x năm đô la $"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


@pytest.mark.parametrize("written", ["2024", "1/2", "10$", "10:30", "12,5kg", "thứ 4", "10k-20k"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(written)


def test_escaped_word_round_trips(normalizer):
    written = 'đường\\a"b'
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
