import importlib

import pytest

from tn.cli import PIPELINES

INPUTS = ["a  b", "#1", "1#", "a #1", 'quote "x"', "back\\slash", "hello , world .", "x\ty", "😀 ok"]
CASES = [(direction, language) for direction, languages in sorted(PIPELINES.items()) for language in sorted(languages)]


@pytest.mark.parametrize("direction, language", CASES)
@pytest.mark.parametrize("text", INPUTS)
def test_every_pipeline_returns_output(shared_pipeline, direction, language, text):
    spec = PIPELINES[direction][language]
    processor = shared_pipeline(getattr(importlib.import_module(spec.module), spec.class_name))
    output = processor.normalize(text)
    assert output.strip()
    result = processor.normalize_with_mapping(text, include_identity=True)
    assert result.output_text == output


@pytest.mark.parametrize("language, written", [
    (language, written)
    for language in ["de", "fr", "ru", "sv"]
    for written in ["3.14", "03.14", "3.13", "3.00"] + (["32.03"] if language != "sv" else [])
])
def test_invalid_numeric_dates_are_not_classified_as_dates(shared_pipeline, language, written):
    spec = PIPELINES["tn"][language]
    normalizer = shared_pipeline(getattr(importlib.import_module(spec.module), spec.class_name))
    parser = normalizer.token_parser()
    parser.parse(normalizer.tag(written))
    assert all(token.name != "date" for token in parser.tokens), normalizer.tag(written)
