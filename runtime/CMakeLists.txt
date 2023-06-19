cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wetextprocessing VERSION 0.1)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_VERBOSE_MAKEFILE OFF)
option(BUILD_TESTING "whether to build unit test" OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base-${CMAKE_CXX_COMPILER_ID}" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FETCHCONTENT_BASE_DIR ${fc_base})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT MSVC)
  # Keep the same with openfst, -fPIC or -fpic
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
else()
  # For windows, please use unicode(3 bytes per chinese char) instead of gbk(2 bytes per chinese char).
  # https://github.com/wenet-e2e/wenet/issues/882#issuecomment-1101246299
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(CMAKE_MACOSX_RPATH 1)
endif()

include(openfst)
include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(utils)
add_subdirectory(processor)
add_subdirectory(bin)

if(BUILD_TESTING)
  include(gtest)
  add_subdirectory(test)
endif()
