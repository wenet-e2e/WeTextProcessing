# Android model assets

The app loads four FST models at runtime (see `MainActivity.java` and
`wetextprocessing.cc`). They are **not** checked into git and must be placed in
this directory before building the APK:

- `zh_tn_tagger.fst`
- `zh_tn_verbalizer.fst`
- `zh_itn_tagger.fst`
- `zh_itn_verbalizer.fst`

## How to generate

From the repository root, generate the TN (text normalization) and ITN
(inverse text normalization) models straight into this folder:

```bash
assets=runtime/android/app/src/main/assets

# TN: produces zh_tn_tagger.fst and zh_tn_verbalizer.fst
python -m tn --language zh --overwrite_cache --cache_dir "$assets"

# ITN: produces zh_itn_tagger.fst and zh_itn_verbalizer.fst
python -m itn --language zh --overwrite_cache --cache_dir "$assets"
```

`pynini` is required to build the models (`pip install pynini importlib_resources`).
On startup `MainActivity.assetsInit()` unpacks these files from assets into the
app's `filesDir`, where the native `Processor` reads them.
