enable_testing()
link_libraries(gtest_main gmock)
include(GoogleTest)

add_executable(string_test string_test.cc)
target_link_libraries(string_test PUBLIC wetext_utils)
gtest_discover_tests(string_test)

if(NOT MSVC)
  # token_parser_test uses the macro to access the private members
  add_executable(token_parser_test token_parser_test.cc)
  target_link_libraries(token_parser_test PUBLIC wetext_processor)
  gtest_discover_tests(token_parser_test)
endif()

add_executable(processor_test processor_test.cc)
target_link_libraries(processor_test PUBLIC wetext_processor)
# Inject an absolute path to the *.fst models and golden data so the test does
# not depend on the working directory.
target_compile_definitions(processor_test PRIVATE
  WETEXT_TN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tn")
gtest_discover_tests(processor_test)

# Host-side JNI regression tests exercise the same native entry points as the
# Android shared library. They are optional when a host-only build has no JDK.
find_package(JNI QUIET)
if(JNI_FOUND)
  add_executable(android_jni_test
    android_jni_test.cc
    ../bindings/android/wetextprocessing.cc)
  target_include_directories(android_jni_test PRIVATE ${JNI_INCLUDE_DIRS})
  target_link_libraries(android_jni_test PUBLIC wetext_processor)
  gtest_discover_tests(android_jni_test)
endif()
