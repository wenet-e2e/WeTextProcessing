include(gflags)
include(glog)

set(CONFIG_FLAGS "")
if(NOT FST_HAVE_BIN)
  if(MSVC)
    set(HAVE_BIN OFF CACHE BOOL "Build the fst binaries" FORCE)
  else()
    set(CONFIG_FLAGS "--disable-bin")
  endif()
endif()

# The original openfst uses GNU Build System to run configure and build.
# So, we use "OpenFST port for Windows" to build openfst with cmake in Windows.
# Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
# To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
set(openfst_BINARY_DIR ${fc_base}/openfst-build CACHE PATH "OpenFST build directory")
set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix CACHE PATH "OpenFST prefix directory")
if(NOT MSVC)
  ExternalProject_Add(openfst
    URL               https://github.com/mjansche/openfst/archive/1.7.2.zip
    URL_HASH          MD5=96656fee440ee2d71006a4900ef9ac00
    PREFIX            ${openfst_PREFIX_DIR}
    SOURCE_DIR        ${openfst_SOURCE_DIR}
    BINARY_DIR        ${openfst_BINARY_DIR}
    CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure ${CONFIG_FLAGS} --prefix=${openfst_PREFIX_DIR} --enable-far
                        "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                        "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                        "LIBS=-lgflags_nothreads -lglog -lpthread"
    COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
    BUILD_COMMAND     make -j$(nproc)
  )
  add_dependencies(openfst gflags glog)
  link_directories(${openfst_PREFIX_DIR}/lib)
else()
  add_compile_options(/W0 /wd4244 /wd4267)
  FetchContent_Declare(openfst
    URL           https://github.com/kkm000/openfst/archive/refs/tags/win/1.7.2.1.zip
    URL_HASH      MD5=50e0dcf010ffd91ab0e4f40bb5b33e8c
    PATCH_COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(openfst)
  add_dependencies(fst gflags glog)
  target_link_libraries(fst PUBLIC gflags_nothreads_static glog)
endif()
include_directories(${openfst_SOURCE_DIR}/src/include)
