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

#include <string>
#include <vector>

#include "gmock/gmock.h"

#define private public

#include "processor/wetext_token_parser.h"

class TokenParserTest : public testing::Test {
 protected:
  wetext::TokenParser* parser;

  virtual void SetUp() {
    parser = new wetext::TokenParser(wetext::ParseType::kTN);
  }

  virtual void TearDown() { delete parser; }
};

TEST_F(TokenParserTest, ReadTest) {
  parser->Load(" ");
  ASSERT_FALSE(parser->Read());
  ASSERT_EQ(parser->ch_, wetext::EOS);
}

TEST_F(TokenParserTest, ParseWSTest) {
  parser->Load(" ");
  ASSERT_FALSE(parser->ParseWs());
  ASSERT_EQ(parser->ch_, wetext::EOS);

  parser->Load("  ");
  ASSERT_FALSE(parser->ParseWs());
  ASSERT_EQ(parser->ch_, wetext::EOS);

  parser->Load("  test");
  ASSERT_TRUE(parser->ParseWs());
  ASSERT_EQ(parser->ch_, "t");
}

TEST_F(TokenParserTest, ParseCharsTest) {
  parser->Load("hello world");
  ASSERT_TRUE(parser->ParseChars("hello"));
  ASSERT_EQ(parser->ch_, " ");

  parser->Load("world");
  ASSERT_FALSE(parser->ParseChars("hello"));
  ASSERT_EQ(parser->ch_, "w");
}

TEST_F(TokenParserTest, ParseKeyTest) {
  parser->Load("key");
  ASSERT_EQ(parser->ParseKey(), "key");

  parser->Load("key ");
  ASSERT_EQ(parser->ParseKey(), "key");
}

TEST_F(TokenParserTest, ParseValueTest) {
  parser->Load("value\"");
  ASSERT_EQ(parser->ParseValue(), "value");
}

TEST_F(TokenParserTest, ParseTest) {
  using testing::ElementsAre;
  using testing::Pair;
  using testing::UnorderedElementsAre;

  std::string input =
      "time { minute: \"零二分\" hour: \"两点\" } char { value: \"走\" }";
  parser->Parse(input);
  std::vector<wetext::Token> tokens = parser->tokens_;
  ASSERT_EQ(tokens.size(), 2);
  ASSERT_EQ(tokens[0].name, "time");
  ASSERT_EQ(tokens[1].name, "char");
  ASSERT_THAT(tokens[0].order, ElementsAre("minute", "hour"));
  ASSERT_THAT(tokens[1].order, ElementsAre("value"));
  ASSERT_THAT(tokens[0].members, UnorderedElementsAre(Pair("minute", "零二分"),
                                                      Pair("hour", "两点")));
  ASSERT_THAT(tokens[1].members, UnorderedElementsAre(Pair("value", "走")));
}

TEST_F(TokenParserTest, ReorderTest) {
  std::string input =
      "time { minute: \"零二分\" hour: \"两点\" } char { value: \"走\" }";
  std::string expected =
      "time { hour: \"两点\" minute: \"零二分\" } char { value: \"走\" }";
  ASSERT_EQ(parser->Reorder(input), expected);
}
