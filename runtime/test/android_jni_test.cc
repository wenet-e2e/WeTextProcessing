// Copyright (c) 2026 WeTextProcessing contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <jni.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "fst/fstlib.h"
#include "gtest/gtest.h"
#include "processor/wetext_processor.h"

namespace wetextprocessing {
extern std::shared_ptr<wetext::Processor> processorTN;
extern std::shared_ptr<wetext::Processor> processorITN;

void init(JNIEnv* env, jobject object, jstring model_dir);
jstring normalize(JNIEnv* env, jobject object, jstring input);
jstring inverse_normalize(JNIEnv* env, jobject object, jstring input);
}  // namespace wetextprocessing

namespace {
struct FakeJniState {
  std::string string_value;
  std::string exception_class;
  std::string exception_message;
  bool exception_pending = false;
  bool throw_on_get_string = false;
};

FakeJniState* current_state = nullptr;

jclass FakeFindClass(JNIEnv*, const char* name) {
  current_state->exception_class = name;
  return reinterpret_cast<jclass>(1);
}

jint FakeThrowNew(JNIEnv*, jclass, const char* message) {
  current_state->exception_message = message;
  current_state->exception_pending = true;
  return JNI_OK;
}

const char* FakeGetStringUTFChars(JNIEnv*, jstring, jboolean*) {
  if (current_state->throw_on_get_string) {
    throw std::runtime_error("simulated JNI string allocation failure");
  }
  return current_state->string_value.c_str();
}

void FakeReleaseStringUTFChars(JNIEnv*, jstring, const char*) {}

jstring FakeNewStringUTF(JNIEnv*, const char*) {
  return reinterpret_cast<jstring>(2);
}

class FakeJNIEnv {
 public:
  explicit FakeJNIEnv(std::string string_value) {
    state_.string_value = std::move(string_value);
    current_state = &state_;
    functions_.FindClass = FakeFindClass;
    functions_.ThrowNew = FakeThrowNew;
    functions_.GetStringUTFChars = FakeGetStringUTFChars;
    functions_.ReleaseStringUTFChars = FakeReleaseStringUTFChars;
    functions_.NewStringUTF = FakeNewStringUTF;
    env_.functions = &functions_;
  }

  ~FakeJNIEnv() { current_state = nullptr; }

  JNIEnv* get() { return &env_; }
  FakeJniState& state() { return state_; }

 private:
  FakeJniState state_;
  JNINativeInterface_ functions_{};
  JNIEnv_ env_{};
};

void ClearProcessors() {
  wetextprocessing::processorTN.reset();
  wetextprocessing::processorITN.reset();
}

std::filesystem::path InstallMinimalProcessors() {
  const auto model_dir = std::filesystem::path(testing::TempDir()) /
                         "wetextprocessing-jni-models";
  std::error_code error;
  std::filesystem::remove_all(model_dir, error);
  std::filesystem::create_directories(model_dir);

  fst::StdVectorFst fst;
  const auto state = fst.AddState();
  fst.SetStart(state);
  fst.SetFinal(state, fst::StdArc::Weight::One());
  if (!fst.Write((model_dir / "zh_tn_tagger.fst").string()) ||
      !fst.Write((model_dir / "zh_tn_verbalizer.fst").string())) {
    return {};
  }

  wetextprocessing::processorTN = std::make_shared<wetext::Processor>(
      (model_dir / "zh_tn_tagger.fst").string(),
      (model_dir / "zh_tn_verbalizer.fst").string());
  wetextprocessing::processorITN = wetextprocessing::processorTN;
  return model_dir;
}

bool WriteSingleArcFst(const std::filesystem::path& path, int input_label,
                       int output_label) {
  fst::StdVectorFst fst;
  const auto start = fst.AddState();
  const auto final = fst.AddState();
  fst.SetStart(start);
  fst.SetFinal(final, fst::StdArc::Weight::One());
  fst.AddArc(start, fst::StdArc(input_label, output_label,
                                fst::StdArc::Weight::One(), final));
  return fst.Write(path.string());
}

std::filesystem::path InstallMalformedProcessors() {
  const auto model_dir = std::filesystem::path(testing::TempDir()) /
                         "wetextprocessing-jni-malformed-models";
  std::error_code error;
  std::filesystem::remove_all(model_dir, error);
  std::filesystem::create_directories(model_dir);

  if (!WriteSingleArcFst(model_dir / "zh_tn_tagger.fst", 'x', 'b') ||
      !WriteSingleArcFst(model_dir / "zh_tn_verbalizer.fst", 'b', 'b') ||
      !WriteSingleArcFst(model_dir / "zh_itn_tagger.fst", 'x', 'b') ||
      !WriteSingleArcFst(model_dir / "zh_itn_verbalizer.fst", 'b', 'b')) {
    return {};
  }

  wetextprocessing::processorTN = std::make_shared<wetext::Processor>(
      (model_dir / "zh_tn_tagger.fst").string(),
      (model_dir / "zh_tn_verbalizer.fst").string());
  wetextprocessing::processorITN = std::make_shared<wetext::Processor>(
      (model_dir / "zh_itn_tagger.fst").string(),
      (model_dir / "zh_itn_verbalizer.fst").string());
  return model_dir;
}

TEST(AndroidJniTest, NormalizeBeforeInitThrowsIllegalState) {
  ClearProcessors();
  FakeJNIEnv fake_env("input");

  EXPECT_EQ(wetextprocessing::normalize(fake_env.get(), nullptr,
                                        reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class,
            "java/lang/IllegalStateException");
}

TEST(AndroidJniTest, FailedInitLeavesProcessingUnavailable) {
  ClearProcessors();
  FakeJNIEnv fake_env("/path/that/does/not/exist");

  wetextprocessing::init(fake_env.get(), nullptr,
                         reinterpret_cast<jstring>(1));
  ASSERT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class,
            "java/lang/IllegalStateException");

  fake_env.state().exception_pending = false;
  fake_env.state().exception_class.clear();
  EXPECT_EQ(wetextprocessing::inverse_normalize(
                fake_env.get(), nullptr, reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class,
            "java/lang/IllegalStateException");
}

TEST(AndroidJniTest, InitExceptionsBecomeJavaIllegalStateExceptions) {
  ClearProcessors();
  FakeJNIEnv fake_env("input");
  fake_env.state().throw_on_get_string = true;

  EXPECT_NO_FATAL_FAILURE(wetextprocessing::init(
      fake_env.get(), nullptr, reinterpret_cast<jstring>(1)));
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class,
            "java/lang/IllegalStateException");
}

TEST(AndroidJniTest, FailedReinitPreservesPublishedProcessors) {
  const auto model_dir = InstallMinimalProcessors();
  ASSERT_FALSE(model_dir.empty());
  FakeJNIEnv fake_env(model_dir.string());

  wetextprocessing::init(fake_env.get(), nullptr,
                         reinterpret_cast<jstring>(1));
  ASSERT_TRUE(fake_env.state().exception_pending);
  ASSERT_NE(wetextprocessing::processorTN, nullptr);
  ASSERT_NE(wetextprocessing::processorITN, nullptr);

  fake_env.state().exception_pending = false;
  fake_env.state().exception_class.clear();
  fake_env.state().string_value.clear();
  EXPECT_EQ(wetextprocessing::normalize(fake_env.get(), nullptr,
                                        reinterpret_cast<jstring>(1)),
            reinterpret_cast<jstring>(2));
  EXPECT_FALSE(fake_env.state().exception_pending);

  ClearProcessors();
  std::error_code error;
  std::filesystem::remove_all(model_dir, error);
}

TEST(AndroidJniTest, ProcessingExceptionsBecomeJavaRuntimeExceptions) {
  const auto model_dir = InstallMinimalProcessors();
  ASSERT_FALSE(model_dir.empty());
  FakeJNIEnv fake_env("input");
  fake_env.state().throw_on_get_string = true;

  EXPECT_EQ(wetextprocessing::normalize(fake_env.get(), nullptr,
                                        reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class, "java/lang/RuntimeException");

  fake_env.state().exception_pending = false;
  fake_env.state().exception_class.clear();
  EXPECT_EQ(wetextprocessing::inverse_normalize(
                fake_env.get(), nullptr, reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class, "java/lang/RuntimeException");

  ClearProcessors();
  std::error_code error;
  std::filesystem::remove_all(model_dir, error);
}

TEST(AndroidJniTest, ProcessingParserExceptionsBecomeJavaRuntimeExceptions) {
  const auto model_dir = InstallMalformedProcessors();
  ASSERT_FALSE(model_dir.empty());
  FakeJNIEnv fake_env("x");

  EXPECT_EQ(wetextprocessing::normalize(fake_env.get(), nullptr,
                                        reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class, "java/lang/RuntimeException");

  fake_env.state().exception_pending = false;
  fake_env.state().exception_class.clear();
  EXPECT_EQ(wetextprocessing::inverse_normalize(
                fake_env.get(), nullptr, reinterpret_cast<jstring>(1)),
            nullptr);
  EXPECT_TRUE(fake_env.state().exception_pending);
  EXPECT_EQ(fake_env.state().exception_class, "java/lang/RuntimeException");

  ClearProcessors();
  std::error_code error;
  std::filesystem::remove_all(model_dir, error);
}
}  // namespace
