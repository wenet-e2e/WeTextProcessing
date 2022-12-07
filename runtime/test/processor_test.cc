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

#include "processor/processor.h"
#include "utils/utf8_string.h"

std::vector<std::pair<std::string, std::string>> parse_test_case(
    const std::string& file_path) {
  const std::string delimiter = "=>";
  std::ifstream file(file_path);

  std::vector<std::pair<std::string, std::string>> test_cases;
  std::string line;
  while (getline(file, line)) {
    CHECK_NE(line.find(delimiter), string::npos);
    std::vector<std::string> arr;
    wenet::split_string(line, delimiter, &arr);
    CHECK_GT(arr.size(), 0);
    CHECK_LE(arr.size(), 2);

    std::string written = wenet::trim(arr[0]);
    std::string spoken = "";
    if (arr.size() == 2) {
      spoken = wenet::trim(arr[1]);
    }
    test_cases.emplace_back(std::make_pair(written, spoken));
  }
  return test_cases;
}

class ProcessorTest
    : public testing::TestWithParam<std::pair<std::string, std::string>> {
 protected:
  wenet::Processor* processor;
  std::string written;
  std::string spoken;

  virtual void SetUp() {
    std::string tagger_path = "../tn/zh_tn_tagger.fst";
    std::string verbalizer_path = "../tn/zh_tn_verbalizer.fst";
    processor = new wenet::Processor(tagger_path, verbalizer_path);
    written = GetParam().first;
    spoken = GetParam().second;
  };

  virtual void TearDown() { delete processor; }
};

TEST_P(ProcessorTest, NormalizeTest) {
  EXPECT_EQ(processor->normalize(written), spoken);
}

std::vector<std::pair<std::string, std::string>> test_cases =
    parse_test_case("../tn/chinese/test/data/normalizer.txt");
INSTANTIATE_TEST_SUITE_P(NormalizeTest, ProcessorTest,
                         testing::ValuesIn(test_cases));
