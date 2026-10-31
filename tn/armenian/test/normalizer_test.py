# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.armenian.normalizer import Normalizer


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
    assert len(CASES) == 134
    # Whitelist expansions keep NeMo's non-breaking spaces verbatim.
    assert CASES[70].values == ("ս.թ.", "սույն\u00a0թվականի")
    assert CASES[84].values == ("", "")


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


@pytest.mark.parametrize("written, kind, members, spoken", [
    ("23", "cardinal", {"integer": "23"}, "քսաներեք"),
    ("8 555", "cardinal", {"integer": "8 555"}, "ութ հազար հինգ հարյուր հիսունհինգ"),
    ("0.2 միլիարդ", "decimal", {"integer_part": "0", "fractional_part": "2", "quantity": "միլիարդ"},
     "զրո ամբողջ երկու միլիարդ"),
    ("4/5", "fraction", {"numerator": "4", "denominator": "5"}, "չորս հինգերորդ"),
    ("1-ին", "ordinal", {"integer": "1", "suffix": "ին"}, "առաջին"),
    ("10-րդ", "ordinal", {"integer": "10", "suffix": "րդ"}, "տասներորդ"),
    ("15:52", "time", {"hours": "15", "minutes": "52"}, "տասնհինգ անց հիսուներկու"),
    ("56.3 կմ²", "measure", {"integer_part": "56", "fractional_part": "3", "units": "կմ²"},
     "հիսունվեց ամբողջ երեք քառակուսի կիլոմետր"),
    ("5-10 դ.", "measure", {"from": "5", "to": "10", "units": "դ."}, "հինգից տասը դար"),
    ("2.5 միլիոն ₸", "money", {
        "integer_part": "2", "fractional_part": "5", "quantity": "միլիոն", "currency": "₸",
    }, "երկու ամբողջ հինգ միլիոն տենգե"),
    ("16000 $", "money", {"integer_part": "16000", "currency": "$"}, "տասնվեց հազար դոլար"),
    ("մ.թ.ա.", "whitelist", {"value": "մ.թ.ա."}, "մեր\u00a0թվարկությունից\u00a0առաջ"),
    ("բարև", "word", {"value": "բարև"}, "բարև"),
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
    ("(5)", "( հինգ )"),
    ("5,", "հինգ ,"),
    ("1 000 000", "մեկ միլիոն"),
    ("1/1-ին", "մեկ առաջին"),
    ("10 միլիարդ $", "տասը միլիարդ դոլար"),
    ("2023 թ.", "երկու հազար քսաներեք թվական"),
])
def test_additional_semantics(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


def test_repeated_keys_prefer_shortest_reading(normalizer):
    assert normalizer.normalize("1 $", nbest=3) == ["մեկ դոլար", "մեկ ամն դոլար", "մեկ ամերիկյան դոլար"]
    assert normalizer.normalize("2-րդ", nbest=2) == ["երկրորդ", "երկուերորդ"]


@pytest.mark.parametrize("written", ["1/2", "15:52", "1 $", "2-րդ", "56.3 կմ²", "մ.թ.ա."])
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
    written = 'ուղի\\a"b'
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
