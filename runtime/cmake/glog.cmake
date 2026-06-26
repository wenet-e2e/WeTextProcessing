if(ANDROID)
  # NDK r30+ ships a host libunwind.so under toolchains/.../prebuilt/linux-x86_64/lib/
  # that glog's find_package(Unwind) picks up and unconditionally links (it gates
  # linking on Unwind_FOUND alone). That x86_64 host library is incompatible with the
  # aarch64 target. WITH_UNWIND=none sets CMAKE_DISABLE_FIND_PACKAGE_Unwind so the
  # lookup is skipped entirely; Clang already provides the unwinder for exceptions.
  set(WITH_UNWIND none CACHE STRING "" FORCE)
endif()

FetchContent_Declare(glog
  URL      https://github.com/google/glog/archive/refs/tags/v0.7.1.zip
  URL_HASH SHA256=c17d85c03ad9630006ef32c7be7c65656aba2e7e2fbfc82226b7e680c771fc88
)
FetchContent_MakeAvailable(glog)
include_directories(${glog_SOURCE_DIR}/src ${glog_BINARY_DIR})
