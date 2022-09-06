add_library(utils STATIC
  utils.cc
  utf8_string.cc
)
target_link_libraries(utils PUBLIC dl fst)
