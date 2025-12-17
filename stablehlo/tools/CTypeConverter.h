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

#ifndef STABLEHLO_TOOLS_CTYPECONVERTER_H_
#define STABLEHLO_TOOLS_CTYPECONVERTER_H_

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir {
namespace stablehlo {

/// Converts MLIR types to C99 types
class CTypeConverter {
public:
  /// Convert an MLIR type to its C99 equivalent
  /// Returns the C type string (e.g., "float*", "int32_t*", "bool*")
  static std::string convertType(Type type, bool useRestrict = true);
  
  /// Convert an element type to its C99 equivalent
  static std::string convertElementType(Type elemType);
  
  /// Get the shape parameter name for a tensor
  static std::string getShapeParamName(StringRef tensorName, int64_t rank);
  
  /// Check if a type is supported for C codegen
  static bool isSupportedType(Type type);
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_CTYPECONVERTER_H_
