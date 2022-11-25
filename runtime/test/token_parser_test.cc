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

#include "processor/token_parser.h"

class TokenParserTest : public testing::Test {
 protected:
  wenet::TokenParser* parser;

  virtual void SetUp() {
    parser = new wenet::TokenParser(wenet::ParseType::kTN);
  }

  virtual void TearDown() { delete parser; }
};

TEST_F(TokenParserTest, ReadTest) {
  parser->load(" ");
  ASSERT_FALSE(parser->read());
  ASSERT_EQ(parser->ch, wenet::EOS);
}

TEST_F(TokenParserTest, ParseWSTest) {
  parser->load(" ");
  ASSERT_FALSE(parser->parse_ws());
  ASSERT_EQ(parser->ch, wenet::EOS);

  parser->load("  ");
  ASSERT_FALSE(parser->parse_ws());
  ASSERT_EQ(parser->ch, wenet::EOS);

  parser->load("  test");
  ASSERT_TRUE(parser->parse_ws());
  ASSERT_EQ(parser->ch, "t");
}

TEST_F(TokenParserTest, ParseCharsTest) {
  parser->load("hello world");
  ASSERT_TRUE(parser->parse_chars("hello"));
  ASSERT_EQ(parser->ch, " ");

  parser->load("world");
  ASSERT_FALSE(parser->parse_chars("hello"));
  ASSERT_EQ(parser->ch, "w");
}

TEST_F(TokenParserTest, ParseKeyTest) {
  parser->load("key");
  ASSERT_EQ(parser->parse_key(), "key");

  parser->load("key ");
  ASSERT_EQ(parser->parse_key(), "key");
}

TEST_F(TokenParserTest, ParseValueTest) {
  parser->load("value\"");
  ASSERT_EQ(parser->parse_value(), "value");
}

TEST_F(TokenParserTest, ParseTest) {
  using testing::ElementsAre;
  using testing::Pair;
  using testing::UnorderedElementsAre;

  std::string input =
      "time { minute: \"零二分\" hour: \"两点\" } char { value: \"走\" }";
  parser->parse(input);
  std::vector<wenet::Token> tokens = parser->tokens;
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
  ASSERT_EQ(parser->reorder(input), expected);
}
