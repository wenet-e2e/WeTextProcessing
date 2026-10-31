from importlib import import_module
from pathlib import Path

import pytest


DATA = Path(__file__).parent / "data"
MODULE = "itn.spanish_english.inverse_normalizer"


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


# Upstream es_en tests load en/..._cased.txt for the cased pipeline, so the
# es_en _cased files never ran there; here every file runs under its own name's
# setting: plain files with cased=False, _cased files with cased=True.
CASES = original_cases(cased=False)
CASED_CASES = original_cases(cased=True)


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(import_module(MODULE).InverseNormalizer)


@pytest.fixture(scope="module")
def cased_normalizer(shared_pipeline):
    return shared_pipeline(import_module(MODULE).InverseNormalizer, cased=True)


def test_original_fixture_counts():
    assert len(CASES) == 749
    assert len(CASED_CASES) == 227


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


@pytest.mark.parametrize("spoken,written", CASED_CASES)
def test_original_cased(cased_normalizer, spoken, written):
    assert cased_normalizer.normalize(spoken) == written
    assert cased_normalizer.verbalize(cased_normalizer.tag(spoken)) == written


@pytest.mark.parametrize("spoken,written", CASES)
def test_original_plain_with_cased_pipeline(cased_normalizer, spoken, written):
    # cased=True only adds capitalized alternatives; lower-case input is unchanged.
    assert cased_normalizer.normalize(spoken) == written


@pytest.mark.parametrize("spoken,written", CASED_CASES)
def test_original_cased_mapping(cased_normalizer, spoken, written):
    result = cased_normalizer.normalize_with_mapping(spoken, include_identity=True)
    assert result.output_text == written
    for mapping in result.mappings:
        assert spoken[mapping.input_start:mapping.input_end] == mapping.input_text
        assert written[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("spoken", ["doce dólares y cinco centavos", "two dollars", "veinte!", "two p m"])
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
    assert [(token.name, token.members) for token in parser.tokens] == [("word", {"value": spoken})]
    assert normalizer.verbalize(tagged) == ""
    assert normalizer.normalize(spoken, nbest=3) == [""]
    results = normalizer.normalize_with_mapping(spoken, nbest=3, include_identity=True)
    assert [result.output_text for result in results] == [""]


@pytest.mark.parametrize("spoken", ['año\\ruta"中文', 'say "twenty three"', "back\\slash dos"])
def test_escaped_word(normalizer, spoken):
    assert normalizer.normalize(spoken) == spoken
    assert normalizer.verbalize(normalizer.tag(spoken)) == spoken


def test_escaped_word_fields(normalizer):
    assert parse(normalizer, 'año\\ruta"中文') == [("word", {"value": 'año\\ruta"中文'})]


def test_empty_input(normalizer):
    assert normalizer.tag("") == ""
    assert normalizer.normalize("") == ""
    assert normalizer.normalize("", nbest=3) == [""]
    assert normalizer.normalize_with_mapping("").output_text == ""


def parse(normalizer, spoken):
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(spoken))
    return [(token.name, token.members) for token in parser.tokens]


def test_raw_spanish_fields(normalizer):
    assert parse(normalizer, "veintitrés") == [("cardinal", {"integer": "veintitrés"})]
    assert parse(normalizer, "doce dólares y cinco centavos") == [
        ("money", {"integer_part": "doce", "currency": "dólares", "cents": "cinco", "minor_currency": "centavos"}),
    ]
    assert parse(normalizer, "primero de enero") == [("date", {"day": "primero", "month": "enero"})]


def test_raw_english_fields(normalizer):
    # English rules are renamed with an "en_" prefix; their fields stay raw.
    assert parse(normalizer, "twenty three") == [("en_cardinal", {"integer": "twenty three"})]
    assert parse(normalizer, "two dollars") == [("en_money", {"value": "two", "currency": "dollars"})]
    assert parse(normalizer, "fifty percent") == [("en_measure", {"value": "fifty", "units": "percent"})]
    assert parse(normalizer, "first") == [("en_ordinal", {"integer": "first"})]
    assert normalizer.normalize("twenty three") == "23"


def test_raw_time_fields(normalizer):
    # An article-less Spanish time takes the English two-digit hour.
    assert parse(normalizer, "dos y media de la tarde") == [
        ("time", {"hours": "dos", "minutes": "y media", "suffix": "de la tarde"}),
    ]
    assert normalizer.normalize("dos y media de la tarde") == "02:30 p.m."
    assert normalizer.normalize("las dos de la tarde") == "las 2:00 p.m."
    assert parse(normalizer, "two p m")[0][0] == "en_time"


def test_raw_whitelist_fields(normalizer):
    assert parse(normalizer, "usted") == [("whitelist", {"value": "usted"})]
    assert parse(normalizer, "doctor") == [("en_whitelist", {"value": "doctor"})]
    # The code-mixed English list has no "DOS", so the Spanish word survives.
    assert normalizer.normalize("dos") == "dos"


def test_code_mixed_sentence(normalizer):
    spoken = "pagué doce dólares and two dollars"
    assert parse(normalizer, spoken) == [
        ("word", {"value": "pagué"}),
        ("money", {"integer_part": "doce", "currency": "dólares"}),
        ("word", {"value": "and"}),
        ("en_money", {"value": "two", "currency": "dollars"}),
    ]
    assert normalizer.normalize(spoken) == "pagué $12 and $2"


def test_cased_option(normalizer, cased_normalizer):
    assert normalizer.normalize("Doce Dólares") == "Doce Dólares"
    assert cased_normalizer.normalize("Doce Dólares") == "$12"


def test_raw_indian_numbering_fields(normalizer):
    assert parse(normalizer, "two lakh five thousand") == [("en_cardinal", {"integer": "two lakh five thousand"})]
    assert normalizer.normalize("two lakh five thousand") == "205000"
    assert normalizer.normalize("three crore") == "30000000"


def test_raw_english_measure_units(normalizer):
    assert parse(normalizer, "nine gigabits per second") == [
        ("en_measure", {"value": "nine", "units": "gigabits per second"}),
    ]
    assert normalizer.normalize("nine gigabits per second") == "9 gbps"
    assert normalizer.normalize("five degrees celsius") == "5 °C"
    assert normalizer.normalize("seventy two degrees fahrenheit") == "72 °F"


def test_raw_english_cased_fields(cased_normalizer):
    # Capitalized raw text stays in the tag; the verbalizer restores the capital on words.
    assert parse(cased_normalizer, "Thirty Billion") == [
        ("en_decimal", {"integer_part": "Thirty", "quantity": "Billion"}),
    ]
    assert cased_normalizer.normalize("Thirty Billion") == "30 Billion"
    assert parse(cased_normalizer, "Two dollars") == [("en_money", {"value": "Two", "currency": "dollars"})]
    assert parse(cased_normalizer, "Seven A M E S T") == [
        ("en_time", {"hour": "Seven", "minute": "", "noon": "A M", "zone": "E S T"}),
    ]
    assert cased_normalizer.normalize("Seven A M E S T") == "07:00 A.M. EST"
    assert parse(cased_normalizer, "Nine GigaBits per second") == [
        ("en_measure", {"value": "Nine", "units": "GigaBits per second"}),
    ]
    assert cased_normalizer.normalize("Nine GigaBits per second") == "9 gbps"
    assert parse(cased_normalizer, "N vidia dot com") == [("en_electronic", {"protocol": "N vidia dot com"})]
    assert cased_normalizer.normalize("W W W. A B C dot com") == "www.ABC.com"
    assert cased_normalizer.normalize("Zero") == "Zero"
