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

#ifndef PROCESSOR_PROCESSOR_H_
#define PROCESSOR_PROCESSOR_H_

#include "fst/extensions/far/farlib.h"

#include "processor/token_parser.h"
#include "utils/log.h"

using fst::FarReader;
using fst::StdArc;
using fst::StringCompiler;

namespace wenet {

class Processor {
 public:
  Processor(const std::string& far_path);
  ~Processor();
  std::string tag(const std::string& input);
  std::string verbalize(const std::string& input);
  std::string normalize(const std::string& input);

 private:
  std::string compose(const std::string& input, const std::string& name);

  TokenParser* parser = nullptr;
  FarReader<StdArc>* reader = nullptr;
  std::shared_ptr<StringCompiler<StdArc>> compiler = nullptr;
};

}  // namespace wenet

#endif  // PROCESSOR_PROCESSOR_H_
