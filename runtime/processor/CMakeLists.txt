add_library(processor STATIC
  processor.cc
  token_parser.cc
)
target_link_libraries(processor PUBLIC utils)
