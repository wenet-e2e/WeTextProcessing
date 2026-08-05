// Copyright (c) 2021 Mobvoi Inc (authors: Xiaoyu Chen)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <jni.h>

#include <exception>
#include <string>

#include "processor/wetext_processor.h"
#include "utils/wetext_flags.h"
#include "utils/wetext_string.h"

namespace wetextprocessing {

std::shared_ptr<wetext::Processor> processorTN;
std::shared_ptr<wetext::Processor> processorITN;

namespace {
class UtfStringChars {
 public:
  UtfStringChars(JNIEnv* env, jstring string)
      : env_(env),
        string_(string),
        chars_(env->GetStringUTFChars(string, nullptr)) {}

  UtfStringChars(const UtfStringChars&) = delete;
  UtfStringChars& operator=(const UtfStringChars&) = delete;

  ~UtfStringChars() {
    if (chars_ != nullptr) {
      env_->ReleaseStringUTFChars(string_, chars_);
    }
  }

  const char* get() const { return chars_; }

 private:
  JNIEnv* env_;
  jstring string_;
  const char* chars_;
};

bool CopyJString(JNIEnv* env, jstring string, std::string* output) {
  UtfStringChars chars(env, string);
  if (chars.get() == nullptr) {
    return false;
  }
  *output = chars.get();
  return true;
}

void ThrowIllegalState(JNIEnv* env, const char* message) {
  jclass exception_class = env->FindClass("java/lang/IllegalStateException");
  if (exception_class != nullptr) {
    env->ThrowNew(exception_class, message);
  }
}
}

void init(JNIEnv* env, jobject, jstring jModelDir) {
  std::string model_dir;
  if (!CopyJString(env, jModelDir, &model_dir)) {
    return;
  }

  try {
    std::string tnTagger = model_dir + "/zh_tn_tagger.fst";
    std::string tnVerbalizer = model_dir + "/zh_tn_verbalizer.fst";
    processorTN = std::make_shared<wetext::Processor>(tnTagger, tnVerbalizer);

    std::string itnTagger = model_dir + "/zh_itn_tagger.fst";
    std::string itnVerbalizer = model_dir + "/zh_itn_verbalizer.fst";
    processorITN = std::make_shared<wetext::Processor>(itnTagger, itnVerbalizer);
  } catch (const std::exception& error) {
    processorTN.reset();
    processorITN.reset();
    ThrowIllegalState(env, error.what());
  }
}

jstring normalize(JNIEnv* env, jobject, jstring input) {
  std::string input_text;
  if (!CopyJString(env, input, &input_text)) {
    return nullptr;
  }
  std::string tagged_text = processorTN->Tag(input_text);
  std::string normalized_text = processorTN->Verbalize(tagged_text);

  return env->NewStringUTF(normalized_text.c_str());
}

jstring inverse_normalize(JNIEnv* env, jobject, jstring input) {
  std::string input_text;
  if (!CopyJString(env, input, &input_text)) {
    return nullptr;
  }
  std::string tagged_text = processorITN->Tag(input_text);
  std::string normalized_text = processorITN->Verbalize(tagged_text);

  return env->NewStringUTF(normalized_text.c_str());
}
}  // namespace wetextprocessing

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("com/wenet/WeTextProcessing/WeTextProcessing");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;)V",
       reinterpret_cast<void*>(wetextprocessing::init)},
      {"normalize", "(Ljava/lang/String;)Ljava/lang/String;",
       reinterpret_cast<void*>(wetextprocessing::normalize)},
      {"inverse_normalize", "(Ljava/lang/String;)Ljava/lang/String;",
       reinterpret_cast<void*>(wetextprocessing::inverse_normalize)}};
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
