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

#include "processor/processor.h"

#include "fst/fstlib.h"

#include "utils/utils.h"

using fst::Fst;
using fst::StdVectorFst;
using fst::StringTokenType;

namespace wenet {

Processor::Processor(const std::string& far_path) {
  reader = FarReader<StdArc>::Open(far_path);
  if (!reader || !reader->Find("tagger") || !reader->Find("verbalizer")) {
    LOG(FATAL) << "Either tagger or verbalizer is missing from the archives.";
  }

  parser = new TokenParser(far_path);
  compiler = std::make_shared<StringCompiler<StdArc>>(StringTokenType::BYTE);
}

Processor::~Processor() {
  delete parser;
  delete reader;
}

std::string Processor::compose(const std::string& input,
                               const std::string& name) {
  CHECK_GT(reader->Find(name), 0);
  const Fst<StdArc>* fst(reader->GetFst());

  StdVectorFst input_fst;
  compiler->operator()(input, &input_fst);

  StdVectorFst lattice;
  fst::Compose(input_fst, *fst, &lattice);

  return shortest_path(lattice);
}

std::string Processor::tag(const std::string& input) {
  return compose(input, "tagger");
}

std::string Processor::verbalize(const std::string& input) {
  return compose(input, "verbalizer");
}

std::string Processor::normalize(const std::string& input) {
  std::string output = tag(input);
  if (output.empty()) {
    return "";
  }
  output = parser->reorder(output);
  return verbalize(output);
}

}  // namespace wenet
