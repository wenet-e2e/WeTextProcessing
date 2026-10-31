from importlib import import_module
from pathlib import Path

import pytest


DATA = Path(__file__).parent / "data"


def original_cases(cased):
    cases = []
    for path in sorted(DATA.glob("test_cases_*.txt")):
        if path.stem.endswith("_cased") != cased:
            continue
        with path.open(encoding="utf-8", newline="") as stream:
            for index, line in enumerate(stream, 1):
                spoken, written = (line[:-1] if line.endswith("\n") else line).split("~")
                cases.append(pytest.param(spoken, written, id="{}-{}".format(path.stem, index)))
    return cases


CASES = original_cases(cased=False)
CASED_CASES = original_cases(cased=True)


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(import_module("itn.spanish.inverse_normalizer").InverseNormalizer)


@pytest.fixture(scope="module")
def cased_normalizer(shared_pipeline):
    return shared_pipeline(import_module("itn.spanish.inverse_normalizer").InverseNormalizer, cased=True)


def test_original_fixture_counts():
    assert len(CASES) == 278
    assert len(CASED_CASES) == 103


@pytest.mark.parametrize("spoken,written", CASES)
def test_original(normalizer, spoken, written):
    assert normalizer.normalize(spoken) == written
    assert normalizer.verbalize(normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken,written", CASES)
def test_original_mapping(normalizer, spoken, written):
    result = normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken,written", CASES)
def test_original_with_cased_pipeline(cased_normalizer, spoken, written):
    # Upstream also runs the lower-cased fixtures through the cased pipeline.
    assert cased_normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", CASED_CASES)
def test_original_cased(cased_normalizer, spoken, written):
    assert cased_normalizer.normalize(spoken) == written
    assert cased_normalizer.verbalize(cased_normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken,written", CASED_CASES)
def test_original_cased_mapping(cased_normalizer, spoken, written):
    result = cased_normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken", ["doce dólares y cinco centavos", "la una y diez", "veinte!", "dos mil"])
def test_nbest_mapping_synchronization(normalizer, spoken):
    outputs = normalizer.normalize(spoken, nbest=3)
    mapped = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in mapped] == outputs
    assert len(outputs) == len(set(outputs))
    assert outputs[0] == normalizer.normalize(spoken)


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
    results = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == [""]
    assert all(mapping.output_text == "" for mapping in results[0].mappings)


def test_escaped_word(normalizer):
    spoken = 'año\\ruta"中文'
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    assert [token.members for token in parser.tokens] == [{"value": spoken}]
    assert normalizer.normalize(spoken) == spoken


def test_empty_input(normalizer):
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


def parse(normalizer, spoken):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    return [(token.name, token.members) for token in parser.tokens]


def test_raw_cardinal_field(normalizer):
    assert parse(normalizer, "veintitrés") == [("cardinal", {"integer": "veintitrés"})]
    assert normalizer.normalize("veintitrés") == "23"


def test_raw_money_fields(normalizer):
    assert parse(normalizer, "doce dólares y cinco centavos") == [
        ("money", {"integer_part": "doce", "currency": "dólares", "cents": "cinco", "minor_currency": "centavos"}),
    ]


def test_raw_time_fields(normalizer):
    assert parse(normalizer, "cuarto para las dos") == [("time", {"minutes": "cuarto para", "hours": "las dos"})]
    assert normalizer.normalize("cuarto para las dos") == "la 1:45"


def test_raw_date_fields(normalizer):
    assert parse(normalizer, "primero de enero") == [("date", {"day": "primero", "month": "enero"})]


def test_cased_option(normalizer, cased_normalizer):
    assert normalizer.normalize("Doce Dólares") == "Doce Dólares"
    assert cased_normalizer.normalize("Doce Dólares") == "$12"
