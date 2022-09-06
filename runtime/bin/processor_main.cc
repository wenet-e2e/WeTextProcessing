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

#include <fstream>
#include <iostream>
#include <string>

#include "processor/processor.h"
#include "processor/token_parser.h"
#include "utils/flags.h"

DEFINE_string(text, "", "input string");
DEFINE_string(file, "", "input file");
DEFINE_string(far, "", "FST archives");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_far.empty()) {
    LOG(FATAL) << "Please provide the FST archives.";
  }
  wenet::Processor processor(FLAGS_far);

  if (!FLAGS_text.empty()) {
    std::string tagged_text = processor.tag(FLAGS_text);
    std::cout << tagged_text << std::endl;
    std::string normalized_text = processor.normalize(FLAGS_text);
    std::cout << normalized_text << std::endl;
  }

  if (!FLAGS_file.empty()) {
    std::ifstream file(FLAGS_file);
    std::string line;
    while (getline(file, line)) {
      std::string tagged_text = processor.tag(line);
      std::cout << tagged_text << std::endl;
      std::string normalized_text = processor.normalize(line);
      std::cout << normalized_text << std::endl;
    }
  }
  return 0;
}
