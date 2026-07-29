import json

import pytest
from pynini import cross

from itn.chinese.inverse_normalizer import InverseNormalizer as ZhInverseNormalizer
from itn.english.inverse_normalizer import InverseNormalizer as EnInverseNormalizer
from itn.japanese.inverse_normalizer import InverseNormalizer as JaInverseNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from tn.japanese.normalizer import Normalizer as JaNormalizer


def _build_small_graphs(processor):
    processor.tagger = cross("input", "tagged")
    processor.verbalizer = cross("tagged", "output")


@pytest.mark.parametrize(
    "normalizer_type,prefix,ordertype,kwargs,expected_config",
    [
        (
            ZhNormalizer,
            "zh_tn",
            "tn",
            {
                "full_to_half": False,
                "remove_erhua": False,
                "remove_interjections": False,
                "remove_puncts": True,
                "tag_oov": True,
                "traditional_to_simple": False,
            },
            {
                "full_to_half": False,
                "remove_erhua": False,
                "remove_interjections": False,
                "remove_puncts": True,
                "tag_oov": True,
                "traditional_to_simple": False,
            },
        ),
        (
            JaNormalizer,
            "ja_tn",
            "tn",
            {
                "full_to_half": False,
                "remove_interjections": True,
                "remove_puncts": True,
                "tag_oov": True,
                "transliterate": True,
            },
            {
                "full_to_half": False,
                "remove_interjections": True,
                "remove_puncts": True,
                "tag_oov": True,
                "transliterate": True,
            },
        ),
        (EnNormalizer, "en_tn", "en_tn", {}, {}),
        (
            ZhInverseNormalizer,
            "zh_itn",
            "itn",
            {
                "enable_0_to_9": True,
                "enable_million": True,
                "enable_standalone_number": False,
                "remove_interjections": False,
            },
            {
                "enable_0_to_9": True,
                "enable_million": True,
                "enable_standalone_number": False,
                "remove_interjections": False,
            },
        ),
        (
            JaInverseNormalizer,
            "ja_itn",
            "itn",
            {
                "enable_0_to_9": True,
                "enable_million": True,
                "enable_standalone_number": False,
                "full_to_half": True,
            },
            {
                "enable_0_to_9": True,
                "enable_million": True,
                "enable_standalone_number": False,
                "full_to_half": True,
            },
        ),
        (EnInverseNormalizer, "en_itn", "itn", {}, {}),
    ],
)
def test_all_pipeline_configs_are_part_of_manifest(
    monkeypatch,
    tmp_path,
    normalizer_type,
    prefix,
    ordertype,
    kwargs,
    expected_config,
):
    monkeypatch.setattr(normalizer_type, "build_tagger_and_verbalizer", _build_small_graphs)
    cache_root = tmp_path / prefix

    normalizer_type(cache_dir=cache_root, **kwargs)

    manifest_path = next(cache_root.glob("**/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["prefix"] == prefix
    assert manifest["ordertype"] == ordertype
    assert manifest["config"] == expected_config
