add_library(wetext_processor STATIC
  wetext_processor.cc
  wetext_token_parser.cc
)

target_link_libraries(wetext_processor PUBLIC fst wetext_utils)

# ----------------------------------------------------------------------------
# C API shared library (wetext_processor_c)
# ----------------------------------------------------------------------------
add_library(wetext_processor_c SHARED
  wetext_processor_c_api.cc
)

target_link_libraries(wetext_processor_c PUBLIC wetext_processor)
