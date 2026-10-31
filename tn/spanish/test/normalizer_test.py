from importlib import import_module
from pathlib import Path

import pytest


DATA = Path(__file__).parent / "data"


def original_cases():
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if path.stem == "test_cases_normalize_with_audio":
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                written, *spoken = (line[:-1] if line.endswith("\n") else line).split("~")
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
    return shared_pipeline(import_module("tn.spanish.normalizer").Normalizer)


@pytest.fixture(scope="module")
def nondeterministic(shared_pipeline):
    return shared_pipeline(import_module("tn.spanish.normalizer").Normalizer, deterministic=False)


def test_original_fixture_counts():
    assert len(CASES) == 537
    assert sum(len(case.values[1]) > 1 for case in CASES) == 7
    assert len(MULTIPLE_CASES) == 29
    assert sum(len(case.values[1]) for case in MULTIPLE_CASES) == 133


@pytest.mark.parametrize("written,spoken", CASES)
def test_original(normalizer, written, spoken):
    # Lines with several spoken forms accept any of them, as upstream does.
    predicted = normalizer.normalize(written)
    assert predicted in spoken
    assert normalizer.verbalize(normalizer.tag(written)) == predicted


@pytest.mark.parametrize("written,spoken", CASES)
def test_original_mapping(normalizer, written, spoken):
    result = normalizer.normalize_with_mapping(written, include_identity=True)
    assert result.output_text in spoken
    for mapping in result.mappings:
        assert written[mapping.input_start:mapping.input_end] == mapping.input_text
        assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("written,spoken", CASES)
def test_original_nondeterministic(nondeterministic, written, spoken):
    candidates = nondeterministic.normalize(written, nbest=500)
    assert any(option in candidates for option in spoken)


@pytest.mark.parametrize("written,expected", MULTIPLE_CASES)
def test_original_multiple_candidates(nondeterministic, written, expected):
    assert set(expected) <= set(nondeterministic.normalize(written, nbest=1000))


@pytest.mark.parametrize("written", ["2/3", "12:30", "$2,20", "1,5 kg", "abc@gmail.com"])
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
    written = 'año\\ruta"中文'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert [token.members for token in parser.tokens] == [{"value": written}]
    assert normalizer.normalize(written) == written


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


@pytest.mark.parametrize("prefix, suffix", [("(", ")"), ("", ","), ("", " !"), ("( ", " )")])
def test_punctuation_spacing_raw_fields_and_mapping(normalizer, prefix, suffix):
    written = prefix + "5" + suffix
    spoken = prefix + "cinco" + suffix
    tagged = normalizer.tag(written)
    parser = normalizer.token_parser()
    parser.parse(tagged)
    expected = [("punctuation", {"value": prefix})] if prefix else []
    expected += [("cardinal", {"integer": "5"}), ("punctuation", {"value": suffix})]
    assert [(token.name, token.members) for token in parser.tokens] == expected
    assert normalizer.normalize(written) == spoken
    assert normalizer.verbalize(tagged) == spoken
    results = normalizer.normalize_with_mapping(written, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == normalizer.normalize(written, nbest=3)
    assert results[0].output_text == spoken
    for result in results:
        for mapping in result.mappings:
            assert written[mapping.input_start:mapping.input_end] == mapping.input_text
            assert result.output_text[mapping.output_start:mapping.output_end] == mapping.output_text


def parse(normalizer, written):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    return [(token.name, token.members) for token in parser.tokens]


def test_raw_cardinal_field(normalizer):
    assert parse(normalizer, "23") == [("cardinal", {"integer": "23"})]
    assert normalizer.normalize("23") == "veintitrés"


def test_raw_time_fields(normalizer):
    tokens = parse(normalizer, "12:30")
    assert len(tokens) == 1 and tokens[0][0] == "time"
    assert set(tokens[0][1].values()) >= {"12", "30"}


def test_raw_money_fields(normalizer):
    tokens = parse(normalizer, "$2,20")
    assert len(tokens) == 1 and tokens[0][0] == "money"
    assert "$" in "".join(tokens[0][1].values())
