/* Copyright 2024 The StableHLO Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "stablehlo/tools/CIndexUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include <sstream>

namespace mlir {
namespace stablehlo {

std::vector<int64_t> CIndexUtils::calculateStrides(ArrayRef<int64_t> shape) {
  std::vector<int64_t> strides;
  if (shape.empty()) return strides;
  
  strides.resize(shape.size());
  int64_t stride = 1;
  // Row-major: last dimension has stride 1, work backwards
  for (int i = shape.size() - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

std::string CIndexUtils::generateLinearIndex(
    StringRef arrayName,
    const std::vector<std::string>& indices,
    const std::vector<int64_t>& strides) {
  if (indices.empty()) return arrayName.str() + "[0]";
  
  std::ostringstream oss;
  oss << arrayName.str() << "[";
  
  bool first = true;
  for (size_t i = 0; i < indices.size() && i < strides.size(); i++) {
    if (!first) oss << " + ";
    oss << indices[i] << " * " << strides[i];
    first = false;
  }
  
  oss << "]";
  return oss.str();
}

std::string CIndexUtils::generateIndexWithShape(
    StringRef arrayName,
    StringRef shapeName,
    const std::vector<std::string>& indices,
    int64_t rank) {
  if (indices.empty()) return arrayName.str() + "[0]";
  
  std::ostringstream oss;
  oss << arrayName.str() << "[";
  
  // Row-major indexing: i0 * shape[1] * shape[2] + i1 * shape[2] + i2
  // For 1D: i0
  // For 2D: i0 * shape[1] + i1
  // For 3D: i0 * shape[1] * shape[2] + i1 * shape[2] + i2
  for (int64_t i = 0; i < rank; i++) {
    if (i > 0) oss << " + ";
    oss << indices[i];
    for (int64_t j = i + 1; j < rank; j++) {
      oss << " * " << shapeName.str() << "[" << j << "]";
    }
  }
  
  oss << "]";
  return oss.str();
}

std::string CIndexUtils::generateLoopVariables(int64_t rank, const std::string& prefix) {
  if (rank == 0) return "";
  
  std::ostringstream oss;
  oss << "int64_t";
  for (int64_t i = 0; i < rank; i++) {
    if (i > 0) oss << ",";
    oss << " " << prefix << i;
  }
  oss << ";";
  return oss.str();
}

std::string CIndexUtils::generateNestedLoops(
    int64_t rank,
    StringRef shapeName,
    const std::string& body,
    const std::string& varPrefix) {
  if (rank == 0) {
    return body;
  }
  
  std::ostringstream oss;
  std::string indent = "  ";
  
  // Generate nested loops
  for (int64_t i = 0; i < rank; i++) {
    for (int64_t j = 0; j < i; j++) {
      oss << indent;
    }
    oss << "for (" << varPrefix << i << " = 0; " 
        << varPrefix << i << " < " << shapeName.str() << "[" << i << "]; " 
        << varPrefix << i << "++) {\n";
    indent += "  ";
  }
  
  // Add body with proper indentation
  oss << indent << body << "\n";
  
  // Close loops
  for (int64_t i = rank - 1; i >= 0; i--) {
    indent = indent.substr(0, indent.length() - 2);
    oss << indent << "}\n";
  }
  
  return oss.str();
}

} // namespace stablehlo
} // namespace mlir
