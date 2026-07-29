from tn.english.rules.cardinal import Cardinal
from tn.english.rules.decimal import Decimal
from tn.english.rules.range import Range
from tn.english.rules.telephone import Telephone
from tn.english.rules.whitelist import WhiteList
from tn.english.normalizer import Normalizer


def test_cardinal_nbest_comes_from_raw_field_verbalizer_paths():
    cardinal = Cardinal(deterministic=False)

    assert [tagged.strip() for tagged in cardinal.tag("256", nbest=8)] == ['cardinal { integer: "256" }']

    outputs = cardinal.normalize("256", nbest=8)
    assert outputs[0] == "two hundred and fifty six"
    assert "two hundred fifty six" in outputs

    mappings = cardinal.normalize_with_mapping("256", nbest=8)
    assert [result.output_text for result in mappings] == outputs
    assert all(result.mappings[0].input_text == "256" for result in mappings)
    assert [result.mappings[0].output_text for result in mappings] == outputs


def test_cardinal_nbest_preserves_digit_and_article_alternatives():
    cardinal = Cardinal(deterministic=False)

    leading_zero_outputs = cardinal.normalize("007", nbest=8)
    assert "oh oh seven" in leading_zero_outputs
    assert "zero zero seven" in leading_zero_outputs

    hundred_outputs = cardinal.normalize("100", nbest=8)
    assert hundred_outputs[0] == "one hundred"
    assert "a hundred" in hundred_outputs


def test_decimal_nbest_preserves_zero_and_oh_alternatives():
    decimal = Decimal(deterministic=False)

    assert set(decimal.normalize(".05", nbest=8)) >= {"point oh five", "point zero five"}
    assert set(decimal.normalize("0.05", nbest=8)) >= {
        "zero point oh five",
        "oh point oh five",
        "zero point zero five",
        "oh point zero five",
    }


def test_range_nbest_preserves_to_and_minus_alternatives():
    outputs = Range(deterministic=False).normalize("2-3", nbest=8)

    assert outputs[0] == "two to three"
    assert "two minus three" in outputs


def test_deterministic_range_keeps_canonical_to_reading():
    rang = Range(deterministic=True)

    assert rang.tag("2-3").strip() == 'range { value: "2-3" }'
    assert rang.normalize("2-3", nbest=8) == ["two to three"]


def test_whitelist_and_telephone_nbest_use_verbalizer_paths():
    whitelist = WhiteList(deterministic=False)
    assert [tagged.strip() for tagged in whitelist.tag("Dr.", nbest=8)] == ['whitelist { name: "Dr." }']
    assert whitelist.normalize("Dr.", nbest=8)[0] == "doctor"

    telephone = Telephone(deterministic=False)
    outputs = telephone.normalize("123-123-5078", nbest=8)
    assert "one two three, one two three, five zero seven eight" in outputs
    assert "one two three, one two three, five oh seven eight" in outputs


def test_deterministic_rules_do_not_expand_spoken_alternatives():
    cardinal = Cardinal(deterministic=True)

    assert cardinal.normalize("007", nbest=8) == ["zero zero seven"]
    assert cardinal.normalize("100", nbest=8) == ["one hundred"]


def test_full_normalizer_electronic_nbest_has_canonical_whitespace(tmp_path):
    normalizer = Normalizer(cache_dir=tmp_path)

    for written in ["cdf1@abc.edu", "http://www.abc.com"]:
        outputs = normalizer.normalize(written, nbest=32)
        assert outputs
        assert all(output == output.strip() for output in outputs)
        assert all("  " not in output for output in outputs)
