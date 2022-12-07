## WeTextProcessing Runtime

1. How to build

``` bash
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```

2. How to use

``` bash
# tn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_tagger.fst
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_verbalizer.fst
$ ./build/bin/processor_main --tagger zh_tn_tagger.fst --verbalizer zh_tn_verbalizer.fst --text "2.5平方电线"

# itn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_tagger.fst
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_verbalizer.fst
$ ./build/bin/processor_main --tagger zh_itn_tagger.fst --verbalizer zh_itn_verbalizer.fst --text "二点五平方电线"
```
