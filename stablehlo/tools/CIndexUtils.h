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

#ifndef STABLEHLO_TOOLS_CINDEXUTILS_H_
#define STABLEHLO_TOOLS_CINDEXUTILS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
namespace stablehlo {

/// Utilities for generating C99 index calculations
class CIndexUtils {
public:
  /// Generate linear index calculation for multi-dimensional array access
  /// Uses row-major (C-style) indexing
  static std::string generateLinearIndex(
      StringRef arrayName, 
      const std::vector<std::string>& indices,
      const std::vector<int64_t>& strides);
  
  /// Generate index calculation with shape array
  static std::string generateIndexWithShape(
      StringRef arrayName,
      StringRef shapeName,
      const std::vector<std::string>& indices,
      int64_t rank);
  
  /// Generate loop variable declarations for nested loops
  static std::string generateLoopVariables(int64_t rank, const std::string& prefix = "i");
  
  /// Generate nested loop structure
  static std::string generateNestedLoops(
      int64_t rank,
      StringRef shapeName,
      const std::string& body,
      const std::string& varPrefix = "i");
  
  /// Calculate strides for a given shape (row-major)
  static std::vector<int64_t> calculateStrides(ArrayRef<int64_t> shape);
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_CINDEXUTILS_H_
