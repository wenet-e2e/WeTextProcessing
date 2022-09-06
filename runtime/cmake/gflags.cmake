set(GFLAGS_NAMESPACE "gflags")

FetchContent_Declare(gflags
  URL      https://github.com/gflags/gflags/archive/v2.2.2.zip
  URL_HASH MD5=ff856ff64757f1381f7da260f79ba79b
)
FetchContent_MakeAvailable(gflags)
include_directories(${gflags_BINARY_DIR}/include)
