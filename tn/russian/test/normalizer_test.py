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

from tn.russian.normalizer import Normalizer


DATA = Path(__file__).parent / "data"
AUDIO = "test_cases_cardinal_normalize_with_audio"
# NeMo's Russian TN is non-deterministic and its tests assert that the expected
# reading is among the candidates (n_tagged=100); the same check is used here.
NBEST = 100
# Two candidates of the (NeMo-untested) audio fixture are misspelled
# ("сорое", "четырый") and cannot be produced by any grammar; they are checked
# against the spelling the fixture evidently intends.
MISSPELLED = {
    "сорое шесть тысяч шестьсот шестьдесят четыре": "сорок шесть тысяч шестьсот шестьдесят четыре",
    "сорое шесть тысяч шестьсот шестьдесят четырый": "сорок шесть тысяч шестьсот шестьдесят четвертый",
}


def original_cases():
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if path.stem == AUDIO:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(written, spoken, id="{}-{}".format(path.stem, index)))
    return cases


def audio_cases():
    cases, written, expected = [], None, []
    with (DATA / (AUDIO + ".txt")).open(encoding="utf-8", newline="") as stream:
        for line in stream:
            line = line[:-1] if line.endswith("\n") else line
            if line.startswith("~"):
                if written is not None:
                    cases.append((written, expected))
                written, expected = line[1:], []
            else:
                expected.append(line)
    cases.append((written, expected))
    return [pytest.param(written, spoken, id="{}-{}".format(written, index))
            for written, expected in cases for index, spoken in enumerate(expected)]


CASES = original_cases()
AUDIO_CASES = audio_cases()


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


def members(normalizer, written):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    return [(token.name, token.members) for token in parser.tokens]


def test_original_fixture_counts():
    assert len(CASES) == 169
    assert len({case.values[0] for case in AUDIO_CASES}) == 21
    assert len(AUDIO_CASES) == 28
    assert CASES[-3].values == ("", "")


@pytest.mark.parametrize("written, spoken", CASES)
def test_original(normalizer, written, spoken):
    assert spoken in normalizer.normalize(written, nbest=NBEST)


@pytest.mark.parametrize("written, spoken", AUDIO_CASES)
def test_original_audio_candidates(normalizer, written, spoken):
    assert MISSPELLED.get(spoken, spoken) in normalizer.normalize(written, nbest=NBEST)


def test_misspelled_audio_candidates_are_not_invented(normalizer):
    outputs = normalizer.normalize("46664", nbest=NBEST)
    assert not set(MISSPELLED) & set(outputs)


@pytest.mark.parametrize("written, spoken", CASES)
def test_original_mapping(normalizer, written, spoken):
    results = normalizer.normalize_with_mapping(written, nbest=NBEST, include_identity=True)
    assert [result.output_text for result in results] == normalizer.normalize(written, nbest=NBEST)
    result = next(result for result in results if result.output_text == spoken)
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert spoken[mapping.output_start:mapping.output_end] == mapping.output_text


def test_tag_then_verbalize_matches_normalize(normalizer):
    for param in CASES:
        written = param.values[0]
        assert normalizer.verbalize(normalizer.tag(written)) == normalizer.normalize(written)


@pytest.mark.parametrize("written", ["2 300", "1,5", "06.03.2006", "13:05", "31 руб.", "a.bc@gmail.com"])
def test_nbest_mapping_synchronization(normalizer, written):
    outputs = normalizer.normalize(written, nbest=3)
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs)) == 3
    assert outputs[0] == normalizer.normalize(written)


@pytest.mark.parametrize("written, expected", [
    ("2 300", [("cardinal", {"integer": "2 300"})]),
    ("-050", [("cardinal", {"negative": "-", "integer": "050"})]),
    ("2-б-34", [("cardinal", {"integer": "2-б-34"})]),
    ("2000-х", [("ordinal", {"integer": "2000-х"})]),
    ("-2,5", [("decimal", {"negative": "-", "integer_part": "2", "fractional_part": "5"})]),
    ("1,23 млрд.", [("decimal", {"integer_part": "1", "fractional_part": "23", "quantity": "млрд."})]),
    ("3 км/ч", [("measure", {"integer": "3", "units": "км/ч"})]),
    ("тест-8", [("measure", {"prefix": "тест", "integer": "8"})]),
    ("6 руб. 35 коп.", [("money", {"integer_part": "6", "currency": "руб."}),
                        ("money", {"integer_part": "35", "currency": "коп."})]),
    ("6/3/1006г.", [("date", {"day": "6", "month": "3", "year": "1006г."})]),
    ("13:05", [("time", {"hours": "13", "minutes": "05"})]),
    ("8-913-983-56-01", [("telephone", {"country_code": "8", "number_part": "913-983-56-01"})]),
    ("a.bc@gmail.com", [("electronic", {"username": "a.bc", "domain": "gmail.com"})]),
    ("м²", [("whitelist", {"value": "м²"})]),
])
def test_raw_fields(normalizer, written, expected):
    assert members(normalizer, written) == expected


def test_time_readings_are_reordered_fields(normalizer):
    tags = normalizer.tag("08:07", nbest=10)
    past = next(tag for tag in tags if "past_hour" in tag)
    assert normalizer.token_parser().reorder(past) == 'time { past_minutes: "07" past_hour: "08" }'
    assert normalizer.verbalize(past) == "семь минут девятого"


@pytest.mark.parametrize("written, spoken", [
    ("2 300", "две тысячи триста"),
    ("21", "двадцать один"),
    ("1000000", "один миллион"),
    ("0,1", "ноль целых одна десятая"),
    ("2,03", "две целых три сотых"),
    ("31 руб.", "тридцать один рубль"),
    ("2 коп.", "две копейки"),
    ("5 USD", "пять долларов"),
    ("2-ая", "вторая"),
    ("02:00", "два часа"),
    ("06.03.2006", "шестое марта две тысячи шестого"),
    ("12 кг", "двенадцать килограм"),
])
def test_best_reading(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken


def test_deterministic_is_not_supported():
    with pytest.raises(ValueError):
        Normalizer(cache_dir=False, deterministic=True)


def test_escaped_word_round_trips(normalizer):
    written = 'путь\\a"b'
    assert members(normalizer, written) == [("word", {"value": written})]
    assert normalizer.normalize(written) == written


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("written", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, written):
    tagged = normalizer.tag(written)
    assert members(normalizer, written) == [("word", {"value": written})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(written, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)
