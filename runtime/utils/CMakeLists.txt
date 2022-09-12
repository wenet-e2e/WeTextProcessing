add_library(utils STATIC
  utils.cc
  utf8_string.cc
)

if(MSVC)
  target_link_libraries(utils PUBLIC fst)
else()
  target_link_libraries(utils PUBLIC dl fst)
endif()

