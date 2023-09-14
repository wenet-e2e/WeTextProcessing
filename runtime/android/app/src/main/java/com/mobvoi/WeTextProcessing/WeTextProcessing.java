package com.mobvoi.WeTextProcessing;

public class WeTextProcessing {

  static {
    System.loadLibrary("wetextprocessing");
  }

  public static native void init(String modelDir);
  public static native String normalize(String input);
  public static native String inverse_normalize(String input);
}
