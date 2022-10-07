include(gflags)
set(WITH_GFLAGS OFF CACHE BOOL "whether build glog with gflags" FORCE)
include(glog)

set(CONFIG_FLAGS "")
if(NOT FST_HAVE_BIN)
  set(CONFIG_FLAGS "--disable-bin")
endif()

# Openfst is compiled with glog/gflags to avoid log and flag conflicts with log and flags in wenet/libtorch.
# To build openfst with gflags and glog, we comment out some vars of {flags, log}.h and flags.cc.
set(openfst_SOURCE_DIR ${fc_base}/openfst-src CACHE PATH "OpenFST source directory")
set(openfst_PREFIX_DIR ${fc_base}/openfst-subbuild/openfst-populate-prefix CACHE PATH "OpenFST prefix directory")
ExternalProject_Add(openfst
  URL               https://github.com/mjansche/openfst/archive/1.7.2.zip
  URL_HASH          MD5=96656fee440ee2d71006a4900ef9ac00
  PREFIX            ${openfst_PREFIX_DIR}
  SOURCE_DIR        ${openfst_SOURCE_DIR}
  CONFIGURE_COMMAND ${openfst_SOURCE_DIR}/configure ${CONFIG_FLAGS} --enable-far --prefix=${openfst_PREFIX_DIR}
                      "CPPFLAGS=-I${gflags_BINARY_DIR}/include -I${glog_SOURCE_DIR}/src -I${glog_BINARY_DIR}"
                      "LDFLAGS=-L${gflags_BINARY_DIR} -L${glog_BINARY_DIR}"
                      "LIBS=-lgflags_nothreads -lglog -lpthread"
  COMMAND           ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/patch/openfst ${openfst_SOURCE_DIR}
  BUILD_COMMAND     make -j$(nproc)
)
add_dependencies(openfst gflags glog)
link_directories(${openfst_PREFIX_DIR}/lib)
include_directories(${openfst_SOURCE_DIR}/src/include)
