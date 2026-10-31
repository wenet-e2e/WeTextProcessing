# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from tn.swedish.normalizer import Normalizer

DATA = Path(__file__).parent / "data"


def original_cases():
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if path.stem == "test_cases_normalize_with_audio":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                written, spoken = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(written, spoken, id="{}-{}".format(path.stem, index)))
    return cases


def multiple_cases():
    cases, written, expected = [], None, []
    with (DATA / "test_cases_normalize_with_audio.txt").open(encoding="utf-8", newline="") as stream:
        for line in stream:
            line = line[:-1] if line.endswith("\n") else line
            if line.startswith("~"):
                if written is not None:
                    cases.append(pytest.param(written, expected, id=written))
                written, expected = line[1:], []
            else:
                expected.append(line)
    if written is not None:
        cases.append(pytest.param(written, expected, id=written))
    return cases


CASES = original_cases()
MULTIPLE_CASES = multiple_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


@pytest.fixture(scope="module")
def nondeterministic(shared_pipeline):
    return shared_pipeline(Normalizer, deterministic=False)


def test_original_fixture_counts():
    assert len(CASES) == 253
    assert len(MULTIPLE_CASES) == 3
    assert sum(len(case.values[1]) for case in MULTIPLE_CASES) == 11
    assert CASES[0].values == ("100", "hundra")


@pytest.mark.parametrize("written, spoken", CASES)
def test_original(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken
    assert normalizer.verbalize(normalizer.tag(written)) == spoken


@pytest.mark.parametrize("written, spoken", CASES)
def test_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text == spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert spoken[mapping.output_start:mapping.output_end] == mapping.output_text


def within_top_candidates(normalizer, written, expected, limit=1000):
    # Widening the n-best list stops early for the common case without changing the limit checked.
    nbest = 10
    while True:
        outputs = set(normalizer.normalize(written, nbest=min(nbest, limit)))
        if expected <= outputs or nbest >= limit:
            return expected <= outputs
        nbest *= 10


@pytest.mark.parametrize("written, spoken", CASES)
def test_original_nondeterministic(nondeterministic, written, spoken):
    assert within_top_candidates(nondeterministic, written, {spoken})


@pytest.mark.parametrize("written, expected", MULTIPLE_CASES)
def test_original_multiple_candidates(nondeterministic, written, expected):
    assert within_top_candidates(nondeterministic, written, set(expected))


@pytest.mark.parametrize("written", ["21", "3. dec. 2022", "$20,50", "kl. 12.34 e.m.", "abc@gmail.com"])
def test_nbest_mapping_synchronization(nondeterministic, written):
    outputs = nondeterministic.normalize(written, nbest=3)
    mapped = nondeterministic.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == nondeterministic.normalize(written)


@pytest.mark.parametrize("written, members", [
    ("-1937", [{"negative": "-", "integer": "1937"}]),
    ("21:a", [{"integer": "21", "suffix": ":a"}]),
    ("1,5 mn", [{"integer_part": "1", "fractional_part": "5", "quantity": "mn"}]),
    ("$20,50", [{"currency_maj": "$", "integer_part": "20", "minor_part": "50"}]),
    ("2 md kr", [{"integer_part": "2", "quantity": "md", "currency_maj": "kr"}]),
    ("2:a jan., 2022 f Kr", [{"day": "2:a", "month": "jan.", "year": "2022", "era": "f Kr"}]),
    ("kl. 12.34 e.m. cet", [{"prefix": "kl.", "hours": "12", "minutes": "34", "suffix": "e.m.", "zone": "cet"}]),
    ("tfn. 08-789 52 25", [{"country_code": "tfn.", "number_part": "08-789 52 25"}]),
    ("2 ½", [{"integer_part": "2", "fraction": "½"}]),
    ("1kg", [{"integer": "1", "units": "kg"}]),
])
def test_raw_fields(normalizer, written, members):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert [token.members for token in parser.tokens] == members


def test_reordered_fields(normalizer):
    assert normalizer.verbalize('date { year: "2022" month: "01" day: "02" }') == "andra januari tjugotjugotvå"
    assert normalizer.verbalize('money { currency_maj: "£" integer_part: "1" minor_part: "20" }') == (
        "ett pund tjugo pence")


def test_deterministic_modes(normalizer, nondeterministic):
    assert normalizer.normalize("21") == "tjugoett"
    assert "tjugoen" not in normalizer.normalize("21", nbest=10)
    assert {"tjugoen", "tjugo ett"} <= set(nondeterministic.normalize("21", nbest=10))


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
    results = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == [""]
    assert all(mapping.output_text == "" for mapping in results[0].mappings)


def test_escaped_word(normalizer):
    written = 'gränd\\väg"中文'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert [token.members for token in parser.tokens] == [{"value": written}]
    assert normalizer.normalize(written) == written


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""
