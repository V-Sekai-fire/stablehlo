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

#ifndef STABLEHLO_TOOLS_COPEMITTERS_H_
#define STABLEHLO_TOOLS_COPEMITTERS_H_

#include "mlir/IR/Operation.h"
#include "stablehlo/dialect/StablehloOps.h"
#include <string>
#include <map>

namespace mlir {
namespace stablehlo {

class CCodeGenerator;

/// Emits C99 code for StableHLO operations
class COpEmitters {
public:
  explicit COpEmitters(CCodeGenerator* generator) : generator(generator) {}
  
  /// Emit C code for an operation
  std::string emitOperation(Operation* op);
  
  /// Emit element-wise operations
  std::string emitAdd(stablehlo::AddOp op);
  std::string emitMultiply(stablehlo::MulOp op);
  std::string emitSubtract(stablehlo::SubtractOp op);
  std::string emitDivide(stablehlo::DivOp op);
  std::string emitMaximum(stablehlo::MaxOp op);
  std::string emitRemainder(stablehlo::RemOp op);
  
  /// Emit constant operation
  std::string emitConstant(stablehlo::ConstantOp op);
  
  /// Emit comparison operations
  std::string emitCompare(stablehlo::CompareOp op);
  
  /// Emit select operation (conditional)
  std::string emitSelect(stablehlo::SelectOp op);
  
  /// Emit logical operations
  std::string emitAnd(stablehlo::AndOp op);
  std::string emitNot(stablehlo::NotOp op);
  
  /// Emit custom call operation (for library functions)
  std::string emitCustomCall(stablehlo::CustomCallOp op);
  
  /// Emit reduction operation
  std::string emitReduce(stablehlo::ReduceOp op);
  
  /// Emit convolution operation
  std::string emitConvolution(stablehlo::ConvolutionOp op);
  
  /// Emit element-wise loop (helper for element-wise ops)
  std::string emitElementWiseLoop(
      Operation* op,
      const std::string& opStr,
      Value lhs,
      Value rhs,
      Value result);
  
  /// Check if operation is supported
  static bool isSupported(Operation* op);

private:
  CCodeGenerator* generator;
  
  /// Get variable name for a value
  std::string getValueName(Value value);
  
  /// Get shape name for a tensor value
  std::string getShapeName(Value value);
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_COPEMITTERS_H_
