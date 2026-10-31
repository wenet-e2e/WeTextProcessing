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

from itn.hindi_english.inverse_normalizer import InverseNormalizer

PACKAGE = Path(__file__).resolve().parent.parent


def original_cases(cased):
    """Upstream runs ``*_cased.txt`` with cased input and every other file with lower-cased input."""

    cases = []
    for path in sorted((Path(__file__).parent / "data").glob("test_cases_*.txt")):
        if path.stem.endswith("_cased") != cased:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                if line.endswith("\n"):
                    line = line[:-1]
                spoken, written = line.split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases(cased=False)
CASED_CASES = original_cases(cased=True)


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer)


@pytest.fixture(scope="module")
def cased_normalizer(shared_pipeline):
    return shared_pipeline(InverseNormalizer, cased=True)


def test_original_fixture_count():
    assert len(CASES) == 416
    assert len(CASED_CASES) == 70
    assert len(CASES) + len(CASED_CASES) == 486


@pytest.mark.parametrize("spoken, written", CASES)
def test_original(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", CASED_CASES)
def test_original_cased(cased_normalizer, spoken, written):
    assert cased_normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, written", CASES)
def test_cased_option_keeps_lower_cased_behaviour(cased_normalizer, spoken, written):
    assert cased_normalizer.normalize(spoken) == written


def _check_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken, written", CASES)
def test_original_mapping(normalizer, spoken, written):
    _check_mapping(normalizer, spoken, written)


@pytest.mark.parametrize("spoken, written", CASED_CASES)
def test_original_cased_mapping(cased_normalizer, spoken, written):
    _check_mapping(cased_normalizer, spoken, written)


def test_tag_then_verbalize_matches_normalize(normalizer, cased_normalizer):
    for pipeline, cases in ((normalizer, CASES), (cased_normalizer, CASED_CASES)):
        for param in cases:
            spoken, written = param.values
            assert pipeline.verbalize(pipeline.tag(spoken)) == written


def _tokens(pipeline, spoken):
    parser = pipeline.token_parser()
    parser.parse(pipeline.tag(spoken))
    return [(token.name, token.members) for token in parser.tokens]


@pytest.mark.parametrize("spoken, tokens, written", [
    # Hindi rules.
    ("एक लाख", [("cardinal", {"integer": "एक लाख"})], "१०००००"),
    ("दसवां", [("ordinal", {"integer": "दसवां"})], "१०वां"),
    ("बारह रुपये पचास पैसे",
     [("money", {"value": "बारह", "currency": "रुपये", "fractional_part": "पचास", "minor_currency": "पैसे"})],
     "₹१२.५०"),
    ("दो सौ मीटर", [("measure", {"value": "दो सौ मीटर"})], "२०० m"),
    # English rules.
    ("twenty three", [("cardinal", {"integer": "twenty three"})], "23"),
    ("twenty dollars", [("money", {"value": "twenty", "currency": "dollars"})], "$20"),
    ("minus sixty six kilograms", [("measure", {"value": "minus sixty six", "units": "kilograms"})], "-66 kg"),
    ("doctor dao", [("whitelist", {"value": "doctor"}), ("word", {"value": "dao"})], "dr. dao"),
    # hi_en glue: Indian numbering, plural temperature units, comma before the year.
    ("twenty one crore ninety eight lakh thirty six thousand five hundred and ninety three",
     [("cardinal", {"integer": "twenty one crore ninety eight lakh thirty six thousand five hundred and ninety three"})],
     "219836593"),
    ("five degrees celsius", [("measure", {"value": "five", "units": "degrees celsius"})], "5 °C"),
    ("the twenty fifth of july twenty twelve",
     [("date", {"day": "the twenty fifth of", "month": "july", "year": "twenty twelve", "preserve_order": "true"})],
     "25 july, 2012"),
    ("july two thousand twelve",
     [("date", {"month": "july", "year": "two thousand twelve", "preserve_order": "true"})], "july 2012"),
    # Code-mixed sentences: each token is tagged by one language.
    ("दस march twenty twenty",
     [("date", {"value": "दस"}), ("date", {"month": "march", "year": "twenty twenty", "preserve_order": "true"})],
     "१० march 2020"),
    ("मैं twenty three साल का हूँ",
     [("word", {"value": "मैं"}), ("cardinal", {"integer": "twenty three"}), ("word", {"value": "साल"}),
      ("word", {"value": "का"}), ("word", {"value": "हूँ"})], "मैं 23 साल का हूँ"),
])
def test_raw_field_tags(normalizer, spoken, tokens, written):
    assert _tokens(normalizer, spoken) == tokens
    assert normalizer.verbalize(normalizer.tag(spoken)) == written
    assert normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken, tokens, written", [
    ("July twenty fifth twenty twelve",
     [("date", {"month": "July", "day": "twenty fifth", "year": "twenty twelve", "preserve_order": "true"})],
     "July 25, 2012"),
    ("The fifteenth of January",
     [("date", {"day": "The fifteenth of", "month": "January", "preserve_order": "true"})], "15 January"),
    ("Twenty three", [("cardinal", {"integer": "Twenty three"})], "23"),
    ("Minus twenty", [("cardinal", {"negative": "Minus", "integer": "twenty"})], "-20"),
    ("Twenty dollars", [("money", {"value": "Twenty", "currency": "dollars"})], "$20"),
    ("Five degrees celsius", [("measure", {"value": "Five", "units": "degrees celsius"})], "5 °C"),
    ("Twenty one crore", [("cardinal", {"integer": "Twenty one crore"})], "210000000"),
    ("Doctor dao", [("whitelist", {"value": "Doctor"}), ("word", {"value": "dao"})], "Dr. dao"),
    # Numbers up to twelve stay words, as in lower-cased input.
    ("Five", [("word", {"value": "Five"})], "Five"),
])
def test_cased_raw_field_tags(cased_normalizer, spoken, tokens, written):
    assert _tokens(cased_normalizer, spoken) == tokens
    assert cased_normalizer.normalize(spoken) == written


def test_lower_cased_pipeline_ignores_capitalized_words(normalizer):
    assert normalizer.normalize("Twenty three") == "Twenty three"
    assert normalizer.normalize("Twenty dollars") == "Twenty dollars"
    assert normalizer.normalize("July twenty fifth twenty twelve") == "July 25th 2012"


@pytest.mark.parametrize("spoken", ["तेईस", "दसवां", "twenty three", "july twenty fifth twenty twelve", "दस march"])
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


@pytest.mark.parametrize("spoken", ['पथ\\a"b', 'path\\a"b'])
def test_escaped_word_round_trips(normalizer, cased_normalizer, spoken):
    for pipeline in (normalizer, cased_normalizer):
        assert [members for _, members in _tokens(pipeline, spoken)] == [{"value": spoken}]
        assert pipeline.normalize(spoken) == spoken


def test_empty_input(normalizer, cased_normalizer):
    for pipeline in (normalizer, cased_normalizer):
        assert pipeline.normalize("") == ""
        assert pipeline.normalize("", nbest=3) == [""]
        assert pipeline.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(normalizer, spoken):
    tagged = normalizer.tag(spoken)
    assert _tokens(normalizer, spoken) == [("word", {"value": spoken})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == [""]
    assert all(mapping.output_text == "" for mapping in mapped[0].mappings)


def test_standalone_from_tn():
    tn_import = re.compile(r"^\s*(from|import)\s+tn\.(hindi|english|hindi_english)\b", re.MULTILINE)
    for path in PACKAGE.rglob("*.py"):
        assert not tn_import.search(path.read_text(encoding="utf-8")), path
