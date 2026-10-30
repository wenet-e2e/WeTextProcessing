from importlib import import_module, util
from pathlib import Path

import pytest


DATA = Path(__file__).parent / "data"


def original_cases():
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases()


def test_original_fixture_count():
    assert len(CASES) == 288


def test_native_pipeline_exists():
    assert util.find_spec("itn.german.inverse_normalizer") is not None


@pytest.fixture(scope="module")
def inverse_normalizer(shared_pipeline):
    return shared_pipeline(import_module("itn.german.inverse_normalizer").InverseNormalizer)


@pytest.mark.parametrize("spoken,written", CASES)
def test_original(inverse_normalizer, spoken, written):
    assert inverse_normalizer.normalize(spoken) == written
    assert inverse_normalizer.verbalize(inverse_normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken,written", CASES)
def test_original_mapping(inverse_normalizer, spoken, written):
    result = inverse_normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken", ["halb zwölf", "eins komma zwei", "zwei euro", "ein ein halb", "dreiundzwanzig"])
def test_nbest_mapping_synchronization(inverse_normalizer, spoken):
    outputs = inverse_normalizer.normalize(spoken, nbest=3)
    results = inverse_normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == inverse_normalizer.normalize(spoken)


@pytest.mark.parametrize("spoken", [" ", "\t", "\n", " \t\n ", "\u00a0"])
def test_whitespace_only_input(inverse_normalizer, spoken):
    tagged = inverse_normalizer.tag(spoken)
    parser = inverse_normalizer.token_parser()
    parser.parse(tagged)
    assert len(parser.tokens) == 1
    assert parser.tokens[0].name == "word"
    assert parser.tokens[0].members == {"value": spoken}
    assert inverse_normalizer.verbalize(tagged) == ""
    assert inverse_normalizer.normalize(spoken, nbest=3) == [""]
    results = inverse_normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == [""]


def test_escaped_word(inverse_normalizer):
    spoken = 'grün\\pfad"中文'
    assert inverse_normalizer.normalize(spoken) == spoken
    parser = inverse_normalizer.token_parser()
    parser.parse(inverse_normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": spoken}]


def test_raw_colloquial_clock(inverse_normalizer):
    parser = inverse_normalizer.token_parser()
    parser.parse(inverse_normalizer.tag("viertel vor zwölf"))
    assert [token.members for token in parser.tokens] == [{"kind": "to", "minutes": "viertel", "hours": "zwölf"}]
    assert inverse_normalizer.normalize("viertel vor zwölf") == "11:45 Uhr"


def test_empty_input(inverse_normalizer):
    assert inverse_normalizer.normalize("") == ""
    assert inverse_normalizer.normalize("", nbest=3) == [""]
    assert inverse_normalizer.normalize_with_mapping("").output_text == ""
