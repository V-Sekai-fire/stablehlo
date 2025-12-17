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

#include "stablehlo/tools/COpEmitters.h"
#include "stablehlo/tools/CCodegen.h"
#include "stablehlo/tools/CTypeConverter.h"
#include "stablehlo/tools/CIndexUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/APFloat.h"
#include <sstream>

namespace mlir {
namespace stablehlo {

std::string COpEmitters::getValueName(Value value) {
  return generator->getValueName(value);
}

std::string COpEmitters::getShapeName(Value value) {
  return generator->getShapeName(value);
}

bool COpEmitters::isSupported(Operation* op) {
  // List of supported operations
  return isa<stablehlo::AddOp>(op) ||
         isa<stablehlo::MulOp>(op) ||
         isa<stablehlo::SubtractOp>(op) ||
         isa<stablehlo::DivOp>(op) ||
         isa<stablehlo::MaxOp>(op) ||
         isa<stablehlo::RemOp>(op) ||
         isa<stablehlo::SelectOp>(op) ||
         isa<stablehlo::ConstantOp>(op) ||
         isa<stablehlo::CompareOp>(op) ||
         isa<stablehlo::ReduceOp>(op) ||
         isa<stablehlo::ConvolutionOp>(op);
}

std::string COpEmitters::emitOperation(Operation* op) {
  if (auto addOp = dyn_cast<stablehlo::AddOp>(op)) {
    return emitAdd(addOp);
  } else if (auto mulOp = dyn_cast<stablehlo::MulOp>(op)) {
    return emitMultiply(mulOp);
  } else if (auto subOp = dyn_cast<stablehlo::SubtractOp>(op)) {
    return emitSubtract(subOp);
  } else if (auto divOp = dyn_cast<stablehlo::DivOp>(op)) {
    return emitDivide(divOp);
  } else if (auto maxOp = dyn_cast<stablehlo::MaxOp>(op)) {
    return emitMaximum(maxOp);
  } else if (auto remOp = dyn_cast<stablehlo::RemOp>(op)) {
    return emitRemainder(remOp);
  } else if (auto selectOp = dyn_cast<stablehlo::SelectOp>(op)) {
    return emitSelect(selectOp);
  } else if (auto constOp = dyn_cast<stablehlo::ConstantOp>(op)) {
    return emitConstant(constOp);
  } else if (auto cmpOp = dyn_cast<stablehlo::CompareOp>(op)) {
    return emitCompare(cmpOp);
  } else if (auto reduceOp = dyn_cast<stablehlo::ReduceOp>(op)) {
    return emitReduce(reduceOp);
  } else if (auto convOp = dyn_cast<stablehlo::ConvolutionOp>(op)) {
    return emitConvolution(convOp);
  }
  
  return ""; // Unsupported
}

std::string COpEmitters::emitAdd(stablehlo::AddOp op) {
  return emitElementWiseLoop(op, "+", op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitMultiply(stablehlo::MulOp op) {
  return emitElementWiseLoop(op, "*", op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitSubtract(stablehlo::SubtractOp op) {
  return emitElementWiseLoop(op, "-", op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitDivide(stablehlo::DivOp op) {
  return emitElementWiseLoop(op, "/", op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitMaximum(stablehlo::MaxOp op) {
  // Use ternary operator: result = (lhs > rhs) ? lhs : rhs
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType) return "";
  
  int64_t rank = resultType.getRank();
  std::string lhsName = getValueName(op.getLhs());
  std::string rhsName = getValueName(op.getRhs());
  std::string resultName = getValueName(op.getResult());
  std::string resultShapeName = getShapeName(op.getResult());
  
  std::ostringstream oss;
  
  // Scope loop variables to avoid name collisions between operations
  oss << "  {\n";
  
  // Generate loop variables
  std::vector<std::string> indices;
  for (int64_t i = 0; i < rank; i++) {
    indices.push_back("i" + std::to_string(i));
  }
  oss << "    " << CIndexUtils::generateLoopVariables(rank) << "\n";
  
  // Generate nested loops
  std::ostringstream body;
  
  // Calculate indices for lhs, rhs, and result
  std::string lhsIdx = CIndexUtils::generateIndexWithShape(
      lhsName, getShapeName(op.getLhs()), indices, rank);
  std::string rhsIdx = CIndexUtils::generateIndexWithShape(
      rhsName, getShapeName(op.getRhs()), indices, rank);
  std::string resultIdx = CIndexUtils::generateIndexWithShape(
      resultName, resultShapeName, indices, rank);
  
  body << resultIdx << " = (" << lhsIdx << " > " << rhsIdx << ") ? " 
       << lhsIdx << " : " << rhsIdx << ";";
  
  // Generate nested loops with proper indentation
  std::string loopCode = CIndexUtils::generateNestedLoops(rank, resultShapeName, body.str());
  std::istringstream loopStream(loopCode);
  std::string line;
  while (std::getline(loopStream, line)) {
    if (!line.empty()) {
      oss << "    " << line << "\n";
    }
  }
  
  oss << "  }\n";
  
  return oss.str();
}

std::string COpEmitters::emitRemainder(stablehlo::RemOp op) {
  return emitElementWiseLoop(op, "%", op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitSelect(stablehlo::SelectOp op) {
  // Generate ternary operator: result = pred ? onTrue : onFalse
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType) return "";
  
  int64_t rank = resultType.getRank();
  std::string predName = getValueName(op.getPred());
  std::string onTrueName = getValueName(op.getOnTrue());
  std::string onFalseName = getValueName(op.getOnFalse());
  std::string resultName = getValueName(op.getResult());
  std::string resultShapeName = getShapeName(op.getResult());
  
  std::ostringstream oss;
  
  // Scope loop variables to avoid name collisions between operations
  oss << "  {\n";
  
  // Generate loop variables
  std::vector<std::string> indices;
  for (int64_t i = 0; i < rank; i++) {
    indices.push_back("i" + std::to_string(i));
  }
  oss << "    " << CIndexUtils::generateLoopVariables(rank) << "\n";
  
  // Generate nested loops
  std::ostringstream body;
  
  // Calculate indices for pred, onTrue, onFalse, and result
  std::string predIdx = CIndexUtils::generateIndexWithShape(
      predName, getShapeName(op.getPred()), indices, rank);
  std::string onTrueIdx = CIndexUtils::generateIndexWithShape(
      onTrueName, getShapeName(op.getOnTrue()), indices, rank);
  std::string onFalseIdx = CIndexUtils::generateIndexWithShape(
      onFalseName, getShapeName(op.getOnFalse()), indices, rank);
  std::string resultIdx = CIndexUtils::generateIndexWithShape(
      resultName, resultShapeName, indices, rank);
  
  body << resultIdx << " = " << predIdx << " ? " << onTrueIdx << " : " << onFalseIdx << ";";
  
  // Generate nested loops with proper indentation
  std::string loopCode = CIndexUtils::generateNestedLoops(rank, resultShapeName, body.str());
  std::istringstream loopStream(loopCode);
  std::string line;
  while (std::getline(loopStream, line)) {
    if (!line.empty()) {
      oss << "    " << line << "\n";
    }
  }
  
  oss << "  }\n";
  
  return oss.str();
}

std::string COpEmitters::emitConstant(stablehlo::ConstantOp op) {
  auto attr = op.getValue();
  auto resultType = op.getResult().getType();
  
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
    // For now, generate a simple constant initialization
    // In a full implementation, we'd extract the actual values from the attribute
    std::ostringstream oss;
    std::string varName = getValueName(op.getResult());
    std::string elemType = CTypeConverter::convertElementType(tensorType.getElementType());
    
    // Check if it's a splat (single value repeated)
    // TODO: When implementing constant initialization, extract and use the splat value:
    //   auto splatValue = denseAttr.getSplatValue<Attribute>();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(attr)) {
      if (denseAttr.isSplat()) {
        oss << "  // Constant (splat): " << varName << "\n";
        // For now, just emit a comment - full implementation would extract value
        oss << "  // TODO: Initialize constant array with splat value\n";
        return oss.str();
      }
    }
    
    oss << "  // Constant tensor: " << varName << "\n";
    oss << "  // TODO: Initialize constant array with values\n";
    return oss.str();
  }
  
  return "";
}

std::string COpEmitters::emitCompare(stablehlo::CompareOp op) {
  // Compare operations return boolean tensors
  std::ostringstream oss;
  std::string lhsName = getValueName(op.getLhs());
  std::string rhsName = getValueName(op.getRhs());
  std::string resultName = getValueName(op.getResult());
  
  auto comparisonDirection = op.getComparisonDirection();
  std::string opStr;
  switch (comparisonDirection) {
    case stablehlo::ComparisonDirection::EQ:
      opStr = "==";
      break;
    case stablehlo::ComparisonDirection::NE:
      opStr = "!=";
      break;
    case stablehlo::ComparisonDirection::GE:
      opStr = ">=";
      break;
    case stablehlo::ComparisonDirection::GT:
      opStr = ">";
      break;
    case stablehlo::ComparisonDirection::LE:
      opStr = "<=";
      break;
    case stablehlo::ComparisonDirection::LT:
      opStr = "<";
      break;
  }
  
  return emitElementWiseLoop(op, opStr, op.getLhs(), op.getRhs(), op.getResult());
}

std::string COpEmitters::emitReduce(stablehlo::ReduceOp op) {
  // Simplified reduction - full implementation would handle all cases
  std::ostringstream oss;
  oss << "  // TODO: Implement reduction operation\n";
  return oss.str();
}

std::string COpEmitters::emitConvolution(stablehlo::ConvolutionOp op) {
  // Simplified convolution - full implementation would handle all cases
  std::ostringstream oss;
  oss << "  // TODO: Implement convolution operation\n";
  return oss.str();
}

std::string COpEmitters::emitElementWiseLoop(
    Operation* op,
    const std::string& opStr,
    Value lhs,
    Value rhs,
    Value result) {
  
  auto resultType = dyn_cast<RankedTensorType>(result.getType());
  if (!resultType) return "";
  
  int64_t rank = resultType.getRank();
  std::string lhsName = getValueName(lhs);
  std::string rhsName = getValueName(rhs);
  std::string resultName = getValueName(result);
  std::string resultShapeName = getShapeName(result);
  
  std::ostringstream oss;
  
  // Scope loop variables to avoid name collisions between operations
  oss << "  {\n";
  
  // Generate loop variables
  std::vector<std::string> indices;
  for (int64_t i = 0; i < rank; i++) {
    indices.push_back("i" + std::to_string(i));
  }
  oss << "    " << CIndexUtils::generateLoopVariables(rank) << "\n";
  
  // Generate nested loops
  std::ostringstream body;
  
  // Calculate indices for lhs, rhs, and result
  std::string lhsIdx = CIndexUtils::generateIndexWithShape(
      lhsName, getShapeName(lhs), indices, rank);
  std::string rhsIdx = CIndexUtils::generateIndexWithShape(
      rhsName, getShapeName(rhs), indices, rank);
  std::string resultIdx = CIndexUtils::generateIndexWithShape(
      resultName, resultShapeName, indices, rank);
  
  body << resultIdx << " = " << lhsIdx << " " << opStr << " " << rhsIdx << ";";
  
  // Generate nested loops with proper indentation (add 2 spaces for block scope)
  std::string loopCode = CIndexUtils::generateNestedLoops(rank, resultShapeName, body.str());
  // Add indentation to each line of the loop code
  std::istringstream loopStream(loopCode);
  std::string line;
  while (std::getline(loopStream, line)) {
    if (!line.empty()) {
      oss << "    " << line << "\n";
    }
  }
  
  oss << "  }\n";
  
  return oss.str();
}

} // namespace stablehlo
} // namespace mlir
