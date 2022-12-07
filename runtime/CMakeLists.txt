cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wetextprocessing VERSION 0.1)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_VERBOSE_MAKEFILE OFF)
option(BUILD_TESTING "whether to build unit test" OFF)

include(FetchContent)
include(ExternalProject)
set(FETCHCONTENT_QUIET OFF)
get_filename_component(fc_base "fc_base" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(FETCHCONTENT_BASE_DIR ${fc_base})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Keep the same with openfst, -fPIC or -fpic
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -fPIC")
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
