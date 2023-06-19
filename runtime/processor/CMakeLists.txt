add_library(processor STATIC
  processor.cc
  token_parser.cc
)
if(MSVC)
  target_link_libraries(processor PUBLIC fst utils)
else()
  target_link_libraries(processor PUBLIC dl fst utils)
endif()
