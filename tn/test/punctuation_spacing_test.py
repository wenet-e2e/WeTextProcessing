import importlib
import re

import pytest

from tn.cli import PIPELINES

INPUTS = ["(5)", "5, 6.", "x (12) y, 7!"]
SPACE_BEFORE_CLOSING = re.compile(r" [,.!?;:)\]}]")
SPACE_AFTER_OPENING = re.compile(r"[(\[{] ")
# The original German fixtures spell attached marks with a space ("20!" -> "zwanzig !").
FIXTURE_SPACED = {"de"}


@pytest.mark.parametrize("language", sorted(set(PIPELINES["tn"]) - FIXTURE_SPACED))
@pytest.mark.parametrize("written", INPUTS)
def test_numbers_keep_attached_punctuation_attached(shared_pipeline, language, written):
    spec = PIPELINES["tn"][language]
    normalizer = shared_pipeline(getattr(importlib.import_module(spec.module), spec.class_name))
    spoken = normalizer.normalize(written)
    assert not SPACE_BEFORE_CLOSING.search(spoken), spoken
    assert not SPACE_AFTER_OPENING.search(spoken), spoken
