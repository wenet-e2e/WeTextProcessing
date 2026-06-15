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

#include "processor/wetext_processor.h"

#include "utils/wetext_log.h"
#include "utils/wetext_string.h"

namespace wetext {

static char32_t UTF8ToCodePoint(const std::string& ch) {
  int len = UTF8CharLength(ch[0]);
  char32_t cp = 0;
  if (len == 1) {
    cp = static_cast<unsigned char>(ch[0]);
  } else if (len == 2) {
    cp = ((static_cast<unsigned char>(ch[0]) & 0x1F) << 6) |
         (static_cast<unsigned char>(ch[1]) & 0x3F);
  } else if (len == 3) {
    cp = ((static_cast<unsigned char>(ch[0]) & 0x0F) << 12) |
         ((static_cast<unsigned char>(ch[1]) & 0x3F) << 6) |
         (static_cast<unsigned char>(ch[2]) & 0x3F);
  } else if (len == 4) {
    cp = ((static_cast<unsigned char>(ch[0]) & 0x07) << 18) |
         ((static_cast<unsigned char>(ch[1]) & 0x3F) << 12) |
         ((static_cast<unsigned char>(ch[2]) & 0x3F) << 6) |
         (static_cast<unsigned char>(ch[3]) & 0x3F);
  }
  return cp;
}

static bool IsKnownChar(char32_t cp) {
  // ASCII printable characters (space to ~)
  if (cp >= 0x0020 && cp <= 0x007E) return true;
  // CJK Unified Ideographs
  if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
  // CJK Unified Ideographs Extension A
  if (cp >= 0x3400 && cp <= 0x4DBF) return true;
  // CJK Compatibility Ideographs
  if (cp >= 0xF900 && cp <= 0xFAFF) return true;
  // CJK Symbols and Punctuation
  if (cp >= 0x3000 && cp <= 0x303F) return true;
  // General Punctuation
  if (cp >= 0x2000 && cp <= 0x206F) return true;
  // Fullwidth forms
  if (cp >= 0xFF00 && cp <= 0xFFEF) return true;
  return false;
}

Processor::Processor(const std::string& tagger_path,
                     const std::string& verbalizer_path) {
  tagger_.reset(StdVectorFst::Read(tagger_path));
  verbalizer_.reset(StdVectorFst::Read(verbalizer_path));
  compiler_ = std::make_shared<StringCompiler<StdArc>>();
  printer_ = std::make_shared<StringPrinter<StdArc>>();

  if (tagger_path.find("zh_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kZH_TN;
  } else if (tagger_path.find("zh_itn_") != tagger_path.npos) {
    parse_type_ = ParseType::kZH_ITN;
  } else if (tagger_path.find("en_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kEN_TN;
  } else if (tagger_path.find("ja_tn_") != tagger_path.npos) {
    parse_type_ = ParseType::kJA_TN;
  } else {
    LOG(FATAL) << "Invalid fst prefix, prefix should contain"
               << " either \"_tn_\" or \"_itn_\".";
  }
}

std::string Processor::ShortestPath(const StdVectorFst& lattice) {
  StdVectorFst shortest_path;
  fst::ShortestPath(lattice, &shortest_path, 1, true);

  std::string output;
  printer_->operator()(shortest_path, &output);
  return output;
}

std::string Processor::Compose(const std::string& input,
                               const StdVectorFst* fst) {
  StdVectorFst input_fst;
  compiler_->operator()(input, &input_fst);

  StdVectorFst lattice;
  fst::Compose(input_fst, *fst, &lattice);
  return ShortestPath(lattice);
}

std::string Processor::Tag(const std::string& input) {
  if (input.empty()) {
    return "";
  }
  return Compose(input, tagger_.get());
}

std::string Processor::Verbalize(const std::string& input) {
  if (input.empty()) {
    return "";
  }
  TokenParser parser(parse_type_);
  std::string output = parser.Reorder(input);

  output = Compose(output, verbalizer_.get());
  output.erase(std::remove(output.begin(), output.end(), '\0'), output.end());
  return output;
}

std::string Processor::TagOOV(const std::string& input) {
  std::vector<std::string> chars;
  SplitUTF8StringToChars(input, &chars);
  std::string output;
  for (const auto& ch : chars) {
    char32_t cp = UTF8ToCodePoint(ch);
    if (IsKnownChar(cp)) {
      output += ch;
    } else {
      output += "<oov>" + ch + "</oov>";
    }
  }
  return output;
}

std::string Processor::Normalize(const std::string& input) {
  std::string output = Verbalize(Tag(input));
  if (parse_type_ == ParseType::kZH_TN &&
      output.find("<oov>") == std::string::npos) {
    output = TagOOV(output);
  }
  return output;
}

}  // namespace wetext
