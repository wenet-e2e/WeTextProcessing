FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  URL_HASH SHA256=40d4ec942217dcc84a9ebe2a68584ada7d4a33a8ee958755763278ea1c5e18ff
)
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
endif()
FetchContent_MakeAvailable(googletest)
