from importlib import import_module
from pathlib import Path

import pytest


DATA = Path(__file__).parent / "data"


def original_cases():
    single, ordinal = [], []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if path.stem == "test_cases_normalize_with_audio":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                param = pytest.param(written, spoken, id="{}-{}".format(path.stem, index))
                (ordinal if path.stem == "test_cases_ordinal" else single).append(param)
    return single, ordinal


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


SINGLE_CASES, ORDINAL_CASES = original_cases()
MULTIPLE_CASES = multiple_cases()


@pytest.fixture(scope="module")
def normalizer():
    return import_module("tn.german.normalizer").Normalizer(cache_dir=False)


@pytest.fixture(scope="module")
def nondeterministic():
    return import_module("tn.german.normalizer").Normalizer(cache_dir=False, deterministic=False)


def test_original_fixture_counts():
    assert len(SINGLE_CASES) == 265
    assert len(ORDINAL_CASES) == 19
    assert len(MULTIPLE_CASES) == 31
    assert sum(len(case.values[1]) for case in MULTIPLE_CASES) == 100


@pytest.mark.parametrize("written,spoken", SINGLE_CASES)
def test_original(normalizer, written, spoken):
    assert normalizer.normalize(written) == spoken
    assert normalizer.verbalize(normalizer.tag(written)) == spoken


@pytest.mark.parametrize("written,spoken", SINGLE_CASES)
def test_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text == spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert spoken[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("written,spoken", SINGLE_CASES)
def test_original_nondeterministic(nondeterministic, written, spoken):
    assert spoken in nondeterministic.normalize(written, nbest=1000)


@pytest.mark.parametrize("written,spoken", ORDINAL_CASES)
def test_original_ordinal_candidates(nondeterministic, written, spoken):
    assert spoken in nondeterministic.normalize(written, nbest=1000)


@pytest.mark.parametrize("written,expected", MULTIPLE_CASES)
def test_original_multiple_candidates(nondeterministic, written, expected):
    assert set(expected) <= set(nondeterministic.normalize(written, nbest=1000))


@pytest.mark.parametrize("written", ["02.03.2003", "11:30 Uhr", "$2,20", "1 1/2", "abc@gmail.com"])
def test_nbest_mapping_synchronization(nondeterministic, written):
    outputs = nondeterministic.normalize(written, nbest=3)
    mapped = nondeterministic.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == nondeterministic.normalize(written)


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
    written = 'grün\\pfad"中文'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert [token.members for token in parser.tokens] == [{"value": written}]
    assert normalizer.normalize(written) == written


def test_raw_clock_fields(normalizer):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag("02:03:01 Uhr"))
    assert [token.members for token in parser.tokens] == [{"hours": "02", "minutes": "03", "seconds": "01"}]


def test_whitelist_modes(normalizer, nondeterministic):
    assert normalizer.normalize("dr.") == "dr."
    assert nondeterministic.normalize("dr.") == "doktor"


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""
