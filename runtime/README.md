## WeTextProcessing Runtime

1. How to build

``` bash
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```

On Windows:
``` bash
$ cmake -DCMAKE_BUILD_TYPE=Release -B build -G "Visual Studio 17 2022" -DBUILD_SHARED_LIBS=0 -DCMAKE_CXX_FLAGS="/ZI"
$ cmake --build build
```

2. How to use

``` bash
# tn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_tagger.fst
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_verbalizer.fst
$ ./build/processor_main --tagger zh_tn_tagger.fst --verbalizer zh_tn_verbalizer.fst --text "2.5平方电线"

# itn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_tagger.fst
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_verbalizer.fst
$ ./build/processor_main --tagger zh_itn_tagger.fst --verbalizer zh_itn_verbalizer.fst --text "二点五平方电线"
```
