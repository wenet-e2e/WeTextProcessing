add_library(wetext_utils STATIC wetext_string.cc)

target_link_libraries(wetext_utils PUBLIC glog)
