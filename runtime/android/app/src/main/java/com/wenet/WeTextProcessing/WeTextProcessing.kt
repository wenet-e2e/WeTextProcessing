package com.wenet.WeTextProcessing

object WeTextProcessing {

    init {
        System.loadLibrary("wetextprocessing")
    }

    @JvmStatic
    external fun init(modelDir: String)

    @JvmStatic
    external fun normalize(input: String): String

    @JvmStatic
    external fun inverse_normalize(input: String): String
}
