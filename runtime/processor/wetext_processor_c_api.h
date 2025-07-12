#ifndef WETEXT_PROCESSOR_C_API_H_
#define WETEXT_PROCESSOR_C_API_H_

#ifdef __cplusplus
extern "C" {
#endif

// Symbol visibility
#if defined(_WIN32) || defined(_WIN64)
  // Symbols are auto-exported on Windows because CMake sets
  // `CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON` for this target. We therefore
  // leave `WETEXT_API` empty to avoid the usual dllexport/dllimport
  // clutter while still allowing the same header to compile elsewhere.
  #define WETEXT_API 
#else
  #define WETEXT_API __attribute__((visibility("default")))
#endif

// Opaque handle to the underlying wetext::Processor C++ object
typedef void* WetextProcessorHandle;

// Create / destroy ---------------------------------------------------------

// Creates a new Processor instance. Returns nullptr on failure.
WETEXT_API WetextProcessorHandle wetext_create_processor(const char* tagger_path,
                                                        const char* verbalizer_path);

// Destroys a Processor instance obtained via wetext_create_processor().
WETEXT_API void wetext_destroy_processor(WetextProcessorHandle handle);

// Processing APIs ----------------------------------------------------------

// The returned C-string is heap allocated and must be released with
// wetext_free_string() once you are done with it.
WETEXT_API const char* wetext_tag(WetextProcessorHandle handle, const char* input);
WETEXT_API const char* wetext_verbalize(WetextProcessorHandle handle, const char* input);
WETEXT_API const char* wetext_normalize(WetextProcessorHandle handle, const char* input);

// Frees a string returned by any of the processing APIs above.
WETEXT_API void wetext_free_string(const char* str);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WETEXT_PROCESSOR_C_API_H_
