enable_testing()
link_libraries(gtest_main gmock)
include(GoogleTest)

add_executable(utf8_string_test utf8_string_test.cc)
target_link_libraries(utf8_string_test PUBLIC utils)
gtest_discover_tests(utf8_string_test)

if(NOT MSVC)
  # token_parser_test uses the macro to access the private members
  add_executable(token_parser_test token_parser_test.cc)
  target_link_libraries(token_parser_test PUBLIC processor)
  gtest_discover_tests(token_parser_test)
endif()

add_executable(processor_test processor_test.cc)
target_link_libraries(processor_test PUBLIC processor)
gtest_discover_tests(processor_test)
