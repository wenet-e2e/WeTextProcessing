import json

import pytest
from pynini import cross

from itn.arabic.inverse_normalizer import InverseNormalizer as ArInverseNormalizer
from itn.armenian.inverse_normalizer import InverseNormalizer as HyInverseNormalizer
from itn.chinese.inverse_normalizer import InverseNormalizer as ZhInverseNormalizer
from itn.english.inverse_normalizer import InverseNormalizer as EnInverseNormalizer
from itn.french.inverse_normalizer import InverseNormalizer as FrInverseNormalizer
from itn.german.inverse_normalizer import InverseNormalizer as DeInverseNormalizer
from itn.hindi.inverse_normalizer import InverseNormalizer as HiInverseNormalizer
from itn.japanese.inverse_normalizer import InverseNormalizer as JaInverseNormalizer
from itn.korean.inverse_normalizer import InverseNormalizer as KoInverseNormalizer
from itn.portuguese.inverse_normalizer import InverseNormalizer as PtInverseNormalizer
from itn.russian.inverse_normalizer import InverseNormalizer as RuInverseNormalizer
from itn.spanish.inverse_normalizer import InverseNormalizer as EsInverseNormalizer
from itn.swedish.inverse_normalizer import InverseNormalizer as SvInverseNormalizer
from itn.vietnamese.inverse_normalizer import InverseNormalizer as ViInverseNormalizer
from tn.arabic.normalizer import Normalizer as ArNormalizer
from tn.armenian.normalizer import Normalizer as HyNormalizer
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from tn.english.normalizer import Normalizer as EnNormalizer
from tn.french.normalizer import Normalizer as FrNormalizer
from tn.german.normalizer import Normalizer as DeNormalizer
from tn.hindi.normalizer import Normalizer as HiNormalizer
from tn.japanese.normalizer import Normalizer as JaNormalizer
from tn.korean.normalizer import Normalizer as KoNormalizer
from tn.portuguese.normalizer import Normalizer as PtNormalizer
from tn.russian.normalizer import Normalizer as RuNormalizer
from tn.spanish.normalizer import Normalizer as EsNormalizer
from tn.swedish.normalizer import Normalizer as SvNormalizer
from tn.vietnamese.normalizer import Normalizer as ViNormalizer


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
        (KoNormalizer, "ko_tn", "tn", {}, {}),
        (DeNormalizer, "de_tn", "tn", {}, {"deterministic": True}),
        (DeNormalizer, "de_tn", "tn", {"deterministic": False}, {"deterministic": False}),
        (HyNormalizer, "hy_tn", "tn", {}, {}),
        (ArNormalizer, "ar_tn", "tn", {}, {}),
        (FrNormalizer, "fr_tn", "tn", {}, {}),
        (PtNormalizer, "pt_tn", "tn", {}, {}),
        (ViNormalizer, "vi_tn", "tn", {}, {}),
        (RuNormalizer, "ru_tn", "tn", {}, {}),
        (SvNormalizer, "sv_tn", "tn", {}, {"deterministic": True}),
        (SvNormalizer, "sv_tn", "tn", {"deterministic": False}, {"deterministic": False}),
        (EsNormalizer, "es_tn", "tn", {}, {"deterministic": True}),
        (EsNormalizer, "es_tn", "tn", {"deterministic": False}, {"deterministic": False}),
        (HiNormalizer, "hi_tn", "tn", {}, {}),
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
        (KoInverseNormalizer, "ko_itn", "itn", {}, {}),
        (DeInverseNormalizer, "de_itn", "itn", {}, {}),
        (HyInverseNormalizer, "hy_itn", "itn", {}, {}),
        (ArInverseNormalizer, "ar_itn", "itn", {}, {}),
        (FrInverseNormalizer, "fr_itn", "itn", {}, {}),
        (PtInverseNormalizer, "pt_itn", "itn", {}, {}),
        (ViInverseNormalizer, "vi_itn", "itn", {}, {}),
        (RuInverseNormalizer, "ru_itn", "itn", {}, {}),
        (SvInverseNormalizer, "sv_itn", "itn", {}, {}),
        (EsInverseNormalizer, "es_itn", "itn", {}, {"cased": False}),
        (EsInverseNormalizer, "es_itn", "itn", {"cased": True}, {"cased": True}),
        (HiInverseNormalizer, "hi_itn", "itn", {}, {}),
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
