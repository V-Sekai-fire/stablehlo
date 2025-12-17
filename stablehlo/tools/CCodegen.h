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

#ifndef STABLEHLO_TOOLS_CCODEGEN_H_
#define STABLEHLO_TOOLS_CCODEGEN_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "stablehlo/tools/CTypeConverter.h"
#include "stablehlo/tools/COpEmitters.h"
#include "stablehlo/tools/CIndexUtils.h"
#include <string>
#include <map>
#include <set>

namespace mlir {
namespace stablehlo {

/// Generates C99 code from StableHLO MLIR
class CCodeGenerator {
public:
  CCodeGenerator() : opEmitters(this), tempCounter(0) {}
  
  /// Generate C99 code for an entire module
  std::string generateModule(ModuleOp module);
  
  /// Generate C99 code for a function
  std::string generateFunction(func::FuncOp funcOp);
  
  /// Generate includes and header
  std::string generateHeader();
  
  /// Get variable name for a value
  std::string getValueName(Value value);
  
  /// Get shape parameter name for a tensor
  std::string getShapeName(Value value);
  
  /// Generate temporary variable name
  std::string generateTempName();
  
  /// Generate memory allocation for a tensor
  std::string generateAllocation(Value value, const std::string& varName);
  
  /// Generate memory deallocation
  std::string generateDeallocation(const std::string& varName);
  
  /// Check if operation is supported
  bool isSupported(Operation* op);

private:
  COpEmitters opEmitters;
  // Use void* as key since Value doesn't have comparison operators
  std::map<void*, std::string> valueNames;
  std::map<void*, std::string> shapeNames;
  int tempCounter;
  
  /// Register value name
  void registerValue(Value value, const std::string& name);
  
  /// Generate function signature
  std::string generateFunctionSignature(func::FuncOp funcOp);
  
  /// Generate function body
  std::string generateFunctionBody(func::FuncOp funcOp);
  
  /// Initialize function arguments
  void initializeFunctionArgs(func::FuncOp funcOp);
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_CCODEGEN_H_
