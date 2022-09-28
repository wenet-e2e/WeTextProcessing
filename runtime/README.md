## WeTextProcessing Runtime

1. How to build

``` bash
$ cmake -B build -DCMAKE_BUILD_TYPE=Release
$ cmake --build build
```

2. How to use

``` bash
# tn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_tn_normalizer.far
$ ./build/bin/processor_main --far zh_tn_normalizer.far --text "2.5平方电线"

# itn usage
$ wget https://github.com/wenet-e2e/WeTextProcessing/releases/download/WeTextProcessing/zh_itn_normalizer.far
$ ./build/bin/processor_main --far zh_itn_normalizer.far --text "二点五平方电线"
```
