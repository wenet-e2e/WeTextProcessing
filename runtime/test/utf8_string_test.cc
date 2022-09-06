// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "gmock/gmock.h"

#include "utils/utf8_string.h"

class StringTest : public testing::Test {};

TEST(StringTest, StringLengthTest) {
  EXPECT_EQ(wenet::string_length("A"), 1);
  EXPECT_EQ(wenet::string_length("À"), 1);
  EXPECT_EQ(wenet::string_length("啊"), 1);
  EXPECT_EQ(wenet::string_length("✐"), 1);
  EXPECT_EQ(wenet::string_length("你好"), 2);
  EXPECT_EQ(wenet::string_length("world"), 5);
}

TEST(StringTest, String2CharsTest) {
  std::vector<std::string> chars;
  wenet::string2chars("你好world", &chars);
  ASSERT_THAT(chars, testing::ElementsAre("你", "好", "w", "o", "r", "l", "d"));
}

TEST(StringTest, TrimTest) {
  ASSERT_EQ(wenet::trim("\thello "), "hello");
  ASSERT_EQ(wenet::trim(" hello\t"), "hello");
}

TEST(StringTest, SplitStringTest) {
  std::vector<std::string> output;
  wenet::split_string("written => spoken", " => ", &output);
  ASSERT_THAT(output, testing::ElementsAre("written", "spoken"));
}
