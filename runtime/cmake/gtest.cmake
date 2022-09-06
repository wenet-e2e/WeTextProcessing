FetchContent_Declare(googletest
  URL      https://github.com/google/googletest/archive/release-1.12.1.zip
  URL_HASH MD5=2648d4138129812611cf6b6b4b497a3b
)
if(MSVC)
  set(gtest_force_shared_crt ON CACHE BOOL "Always use msvcrt.dll" FORCE)
endif()
FetchContent_MakeAvailable(googletest)
