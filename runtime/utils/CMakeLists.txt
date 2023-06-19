add_library(utils STATIC string.cc)

target_link_libraries(utils PUBLIC glog)
