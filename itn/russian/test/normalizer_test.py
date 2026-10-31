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

from itn.russian.inverse_normalizer import InverseNormalizer


DATA = Path(__file__).parent / "data"
# test_cases_ordinal_hard.txt is not run by NeMo's own tests and contradicts
# test_cases_ordinal.txt ("первый~1-й" versus "я в первый раз~я в первый раз"),
# so its marked endings are checked among the candidates instead of first.
HARD = "test_cases_ordinal_hard"
NBEST = 100


def read_cases(hard):
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if (path.stem == HARD) != hard:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = read_cases(hard=False)
HARD_CASES = read_cases(hard=True)


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


def members(normalizer, spoken):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    return [(token.name, token.members) for token in parser.tokens]


def test_original_fixture_counts():
    assert len(CASES) == 169
    assert len(HARD_CASES) == 57
    assert ("x ", "x") in [case.values for case in CASES]
    assert ("", "") in [case.values for case in CASES]


@pytest.mark.parametrize("spoken, written", CASES)
def test_original(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", HARD_CASES)
def test_original_hard_candidates(normalizer, spoken, written):
    assert written in normalizer.normalize(spoken, nbest=NBEST)


@pytest.mark.parametrize("spoken, written", CASES)
def test_original_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES + HARD_CASES:
        spoken = param.values[0]
        assert normalizer.verbalize(normalizer.tag(spoken)) == normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken", [
    "минус шестьдесят",
    "восьмого",
    "двадцать первые",
    "три километра в час",
    "шесть рублей тридцать пять копеек",
    "семь минут девятого",
    'путь\\a"b',
])
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


@pytest.mark.parametrize("spoken, expected", [
    ("минус шестьдесят", [("cardinal", {"negative": "минус", "integer": "шестьдесят"})]),
    ("двадцать первые", [("ordinal", {"integer": "двадцать первые"})]),
    ("минус две целых пять десятых", [("decimal", {
        "negative": "минус", "integer_part": "две целых", "fractional_part": "пять десятых"})]),
    ("одна целая двадцать три сотых миллиарда", [("decimal", {
        "integer_part": "одна целая", "fractional_part": "двадцать три сотых", "quantity": "миллиарда"})]),
    ("три километра в час", [("measure", {"integer": "три", "units": "километра в час"})]),
    ("шесть рублей тридцать пять копеек", [
        ("money", {"integer_part": "шесть", "currency": "рублей"}),
        ("money", {"integer_part": "тридцать пять", "currency": "копеек"})]),
    ("ноль второго пятого две тысячи девятого года", [("date", {
        "day": "ноль второго", "month": "пятого", "year": "две тысячи девятого года"})]),
    ("тринадцать часов пять минут", [("time", {"hours": "тринадцать часов", "minutes": "пять минут"})]),
    ("без пятнадцати минут шесть", [("time", {"to_minutes": "без пятнадцати минут", "to_hour": "шесть"})]),
    ("восемь девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один", [("telephone", {
        "country_code": "восемь",
        "number_part": "девятьсот тринадцать девятьсот восемьдесят три пятьдесят шесть ноль один"})]),
    ("эй собака хот мейл точка ди и", [("electronic", {"username": "эй", "domain": "хот мейл точка ди и"})]),
    ("жилой комплекс Принцип", [("whitelist", {"value": "жилой комплекс"}), ("word", {"value": "Принцип"})]),
    ("девять", [("word", {"value": "девять"})]),
])
def test_raw_fields(normalizer, spoken, expected):
    assert members(normalizer, spoken) == expected


def test_time_readings_are_reordered_fields(normalizer):
    tagged = normalizer.tag("семь минут девятого")
    assert members(normalizer, "семь минут девятого") == [
        ("time", {"past_minutes": "семь минут", "past_hour": "девятого"})]
    assert normalizer.token_parser().reorder(tagged) == 'time { past_hour: "девятого" past_minutes: "семь минут" }'
    assert normalizer.verbalize(tagged) == "08:07"


@pytest.mark.parametrize("spoken, written", [
    ("две тысячи триста", "2300"),
    ("минус одна целая пять десятых", "-1,5"),
    ("тридцать один рубль", "31 руб."),
    ("сто один рубль", "101 руб."),
    ("одиннадцать рублей", "11 руб."),
    ("пять долларов", "5 USD"),
    ("пять минут третьего", "02:05"),
    ("шестое марта две тысячи шестого", "06.03.2006"),
    ("двенадцать килограм", "12 кг"),
])
def test_best_reading(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, top, candidate", [
    # Lone digits stay words; the digit and marked readings are alternatives.
    ("девять", "девять", "9"),
    ("восьмого", "восьмого", "8-го"),
    ("двадцать первые", "21", "21-е"),
    ("семь минут девятого", "08:07", "20:07"),
])
def test_alternatives(normalizer, spoken, top, candidate):
    outputs = normalizer.normalize(spoken, nbest=10)
    assert outputs[0] == top
    assert candidate in outputs


def test_escaped_word_round_trips(normalizer):
    spoken = 'путь\\a"b'
    assert members(normalizer, spoken) == [("word", {"value": spoken})]
    assert normalizer.normalize(spoken) == spoken


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, spoken):
    tagged = normalizer.tag(spoken)
    assert members(normalizer, spoken) == [("word", {"value": spoken})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
