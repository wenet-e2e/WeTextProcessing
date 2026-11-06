"""Pin the released-version regressions reported in issue #403.

English sentence token boundaries and raw-field measure verbalization already
fix these inputs. Keep both behaviors covered without changing rule weights.
"""

import pytest

from tn.english.normalizer import Normalizer


@pytest.fixture(scope="module")
def normalizer(shared_pipeline):
    return shared_pipeline(Normalizer)


@pytest.mark.parametrize("word", ["day", "hour", "second", "month", "minute", "page", "week", "year", "person"])
def test_per_prose_word_is_not_a_unit_prefix(normalizer, word):
    source = "It happens per {}.".format(word)
    assert normalizer.normalize(source) == source
    # Check classification too: a partial unit must never beat a whole word.
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(source))
    assert all(token.name != "measure" for token in parser.tokens)
    assert normalizer.normalize_with_mapping(source, include_identity=True).output_text == source


@pytest.mark.parametrize("source,expected", [
    ("price per kg", "price per kilogram"),
    ("price per km", "price per kilometer"),
    ("price per min", "price per minute"),
    ("price /kg", "price per kilogram"),
    ("price per kg.", "price per kilogram."),
    ("price per kg, please", "price per kilogram, please"),
    ("5 km per hour", "five kilometers per hour"),
    ("10 GB per second", "ten gigabytes per second"),
    ("The CI system does about 25 million of them per day.",
     "The CI system does about twenty five million of them per day."),
])
def test_rate_forms_and_sentences(normalizer, source, expected):
    assert normalizer.normalize(source) == expected
    result = normalizer.normalize_with_mapping(source, include_identity=True)
    assert result.output_text == expected
    for mapping in result.mappings:
        assert source[mapping.input_start:mapping.input_end] == mapping.input_text
        assert expected[mapping.output_start:mapping.output_end] == mapping.output_text


@pytest.mark.parametrize("source,expected", [("per kg", "per kilogram"), ("per km", "per kilometer")])
def test_unit_only_tags_have_no_spoken_dash(normalizer, source, expected):
    tagged = normalizer.tag(source)
    assert normalizer.verbalize(tagged) == expected
    outputs = normalizer.normalize(source, nbest=10)
    assert outputs[0] == expected
    assert all("-" not in output for output in outputs)
    assert [result.output_text for result in normalizer.normalize_with_mapping(source, nbest=10)] == outputs


def test_slash_rate_keeps_all_existing_unit_readings(normalizer):
    # The h table has more than one pronunciation; none may split off a suffix
    # or acquire the old unit-only "-" sentinel in the spoken output.
    outputs = normalizer.normalize("5 km/h", nbest=10)
    assert "five kilometers per hour" in outputs
    assert all(not output.startswith("-") and " - " not in output for output in outputs)
