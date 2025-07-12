#include "processor/wetext_processor_c_api.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>

#include "processor/wetext_processor.h"

using wetext::Processor;

// Utility ------------------------------------------------------------------
namespace {
// Copies an std::string into a newly allocated C string that the caller must
// free via wetext_free_string().
const char* CopyToCString(const std::string& str) {
  char* out = new char[str.size() + 1];
  std::memcpy(out, str.c_str(), str.size() + 1);
  return out;
}
}  // namespace

// Public API ---------------------------------------------------------------

WetextProcessorHandle wetext_create_processor(const char* tagger_path,
                                             const char* verbalizer_path) {
  if (!tagger_path || !verbalizer_path) {
    return nullptr;
  }
  try {
    Processor* proc = new Processor(tagger_path, verbalizer_path);
    return static_cast<WetextProcessorHandle>(proc);
  } catch (...) {
    return nullptr;
  }
}

void wetext_destroy_processor(WetextProcessorHandle handle) {
  if (!handle) return;
  Processor* proc = static_cast<Processor*>(handle);
  delete proc;
}

#define WETEXT_RETURN_STRING(expr)                    \
  if (!handle || !input) return nullptr;              \
  Processor* proc = static_cast<Processor*>(handle);  \
  std::string result = (expr);                        \
  return CopyToCString(result);

const char* wetext_tag(WetextProcessorHandle handle, const char* input) {
  WETEXT_RETURN_STRING(proc->Tag(input));
}

const char* wetext_verbalize(WetextProcessorHandle handle, const char* input) {
  WETEXT_RETURN_STRING(proc->Verbalize(input));
}

const char* wetext_normalize(WetextProcessorHandle handle, const char* input) {
  WETEXT_RETURN_STRING(proc->Normalize(input));
}

void wetext_free_string(const char* str) {
  delete[] str;
}
