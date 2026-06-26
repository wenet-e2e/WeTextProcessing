cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(wetextprocessing VERSION 0.1)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_TESTING "whether to build unit test" OFF)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # For windows, please use unicode(3 bytes per chinese char) instead of gbk(2 bytes per chinese char).
  # https://github.com/wenet-e2e/wenet/issues/882#issuecomment-1101246299
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(CMAKE_MACOSX_RPATH 1)
endif()

if(ANDROID)
  set(BUILD_SHARED_LIBS OFF)
endif()

include(openfst)
include(glog)
include(gflags)
include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

add_subdirectory(utils)
add_subdirectory(processor)

if(ANDROID)
  # JNI shared library; only meaningful when cross-compiling with the NDK.
  add_subdirectory(bindings/android)
else()
  # Command-line tools are host-only; skip them for Android.
  add_subdirectory(bin)
endif()

if(BUILD_TESTING)
  include(gtest)
  add_subdirectory(test)
endif()
