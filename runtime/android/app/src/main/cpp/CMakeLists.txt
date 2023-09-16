cmake_minimum_required(VERSION 3.4.1)
set(TARGET wetextprocessing)
project(${TARGET} CXX)
set(CMAKE_CXX_STANDARD 14)
include(ExternalProject)

set(CMAKE_VERBOSE_MAKEFILE on)
set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
string(REPLACE "-Wl,--exclude-libs,libgcc_real.a" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

include(openfst)

include_directories(
  ${CMAKE_SOURCE_DIR}
)

add_subdirectory(utils)
add_subdirectory(processor)

link_libraries(wetext_processor android)
add_library(${TARGET} SHARED wetextprocessing.cc)
