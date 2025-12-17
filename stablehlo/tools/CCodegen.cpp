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

#include "stablehlo/tools/CCodegen.h"
#include "stablehlo/tools/CTypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Block.h"
#include <sstream>

namespace mlir {
namespace stablehlo {

std::string CCodeGenerator::generateHeader() {
  std::ostringstream oss;
  oss << "// Generated C99 code from StableHLO\n";
  oss << "// Compile with: gcc -std=c99 ...\n\n";
  oss << "#include <stdint.h>\n";
  oss << "#include <stdlib.h>\n";
  oss << "#include <stdbool.h>\n\n";
  return oss.str();
}

std::string CCodeGenerator::generateModule(ModuleOp module) {
  std::ostringstream oss;
  oss << generateHeader();
  
  // Generate code for each function
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    oss << generateFunction(funcOp) << "\n";
  }
  
  return oss.str();
}

std::string CCodeGenerator::generateFunction(func::FuncOp funcOp) {
  std::ostringstream oss;
  
  // Clear value names for this function
  valueNames.clear();
  shapeNames.clear();
  tempCounter = 0;
  
  // Initialize function arguments (must happen before signature generation)
  initializeFunctionArgs(funcOp);
  
  // Generate function signature (this may register result values)
  oss << generateFunctionSignature(funcOp) << " {\n";
  
  // Generate function body
  oss << generateFunctionBody(funcOp);
  
  oss << "}\n";
  return oss.str();
}

std::string CCodeGenerator::generateFunctionSignature(func::FuncOp funcOp) {
  std::ostringstream oss;
  auto funcType = funcOp.getFunctionType();
  
  // Return type - always void since results are passed as output parameters
  oss << "void";
  
  // Avoid conflict with C's main function
  std::string funcName = funcOp.getName().str();
  if (funcName == "main") {
    funcName = "stablehlo_main";
  }
  
  oss << " " << funcName << "(";
  
  // Input parameters
  bool first = true;
  for (unsigned i = 0; i < funcType.getNumInputs(); i++) {
    if (!first) oss << ", ";
    first = false;
    
    auto inputType = funcType.getInput(i);
    auto arg = funcOp.getBody().front().getArgument(i);
    std::string argName = getValueName(arg);
    
    oss << CTypeConverter::convertType(inputType) << " " << argName;
    
    // Add shape parameter for tensors
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
      int64_t rank = tensorType.getRank();
      if (rank > 0) {
        oss << ", int64_t " << getShapeName(arg) << "[" << rank << "]";
      }
    }
  }
  
  // Output parameters (if multiple results or single result passed by reference)
  if (funcType.getNumResults() > 0) {
    for (unsigned i = 0; i < funcType.getNumResults(); i++) {
      if (!first) oss << ", ";
      first = false;
      
      auto resultType = funcType.getResult(i);
      std::string resultName = "result" + (funcType.getNumResults() > 1 ? std::to_string(i) : "");
      
      // Note: Result value registration happens in generateFunctionBody
      // when we encounter the return operation
      
      oss << CTypeConverter::convertType(resultType) << " " << resultName;
      
      // Add shape parameter for tensors
      if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
        int64_t rank = tensorType.getRank();
        if (rank > 0) {
          oss << ", int64_t " << resultName << "_shape[" << rank << "]";
        }
      }
    }
  }
  
  oss << ")";
  return oss.str();
}

std::string CCodeGenerator::generateFunctionBody(func::FuncOp funcOp) {
  std::ostringstream oss;
  
  Block& entryBlock = funcOp.getBody().front();
  
  // Track allocations for cleanup
  std::vector<std::string> allocations;
  
  // Track unsupported operations for error reporting
  std::vector<Operation*> unsupportedOps;
  
  // First pass: register all result values to ensure proper naming
  // This ensures that when we generate code, all values have names
  for (auto& op : entryBlock) {
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      // Handle return operation - register result values
      if (auto returnOp = dyn_cast<func::ReturnOp>(op)) {
        auto funcType = funcOp.getFunctionType();
        for (unsigned i = 0; i < returnOp.getNumOperands() && i < funcType.getNumResults(); i++) {
          auto resultValue = returnOp.getOperand(i);
          std::string resultName = "result" + (funcType.getNumResults() > 1 ? std::to_string(i) : "");
          registerValue(resultValue, resultName);
          
          // Register shape name
          auto resultType = funcType.getResult(i);
          if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
            int64_t rank = tensorType.getRank();
            if (rank > 0) {
              std::string shapeName = resultName + "_shape";
              void* valuePtr = resultValue.getAsOpaquePointer();
              shapeNames[valuePtr] = shapeName;
            }
          }
        }
      }
      continue;
    }
    
    // Register result names for operations
    for (unsigned i = 0; i < op.getNumResults(); i++) {
      auto result = op.getResult(i);
      void* valuePtr = result.getAsOpaquePointer();
      if (valueNames.find(valuePtr) == valueNames.end()) {
        std::string name = generateTempName();
        registerValue(result, name);
      }
    }
  }
  
  // Second pass: generate code for each operation
  for (auto& op : entryBlock) {
    // Skip terminator
    if (op.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }
    
    // Check if operation is supported
    if (!isSupported(&op)) {
      unsupportedOps.push_back(&op);
      continue;
    }
    
    // Allocate memory for operation results if needed (before generating code)
    for (unsigned i = 0; i < op.getNumResults(); i++) {
      auto result = op.getResult(i);
      if (auto tensorType = dyn_cast<RankedTensorType>(result.getType())) {
        std::string varName = getValueName(result);
        std::string allocCode = generateAllocation(result, varName);
        if (!allocCode.empty()) {
          oss << allocCode << "\n";
          allocations.push_back(varName);
        }
      }
    }
    
    // Generate code for the operation (after allocation)
    std::string opCode = opEmitters.emitOperation(&op);
    if (!opCode.empty()) {
      oss << opCode << "\n";
    }
  }
  
  // Report unsupported operations
  if (!unsupportedOps.empty()) {
    oss << "\n  // ERROR: The following operations are not supported:\n";
    for (auto* op : unsupportedOps) {
      std::string opName = op->getName().getStringRef().str();
      oss << "  //   - " << opName << "\n";
    }
    oss << "  // Please use only supported operations: add, multiply, subtract, divide, constant, compare\n";
  }
  
  // Generate cleanup code (deallocations) - in reverse order
  for (auto it = allocations.rbegin(); it != allocations.rend(); ++it) {
    oss << generateDeallocation(*it) << "\n";
  }
  
  return oss.str();
}

std::string CCodeGenerator::generateAllocation(Value value, const std::string& varName) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType) return "";
  
  // Skip allocation if it's a function argument or return value
  // They should be provided by the caller
  if (isa<BlockArgument>(value)) {
    return ""; // Function argument, no allocation needed
  }
  
  // Skip allocation if it's a function return value (passed as parameter)
  // Check if this value is used as a return value
  bool isReturnValue = false;
  for (auto& use : value.getUses()) {
    if (isa<func::ReturnOp>(use.getOwner())) {
      isReturnValue = true;
      break;
    }
  }
  if (isReturnValue) {
    return ""; // Return value, provided by caller
  }
  
  std::ostringstream oss;
  std::string elemType = CTypeConverter::convertElementType(tensorType.getElementType());
  std::string shapeName = getShapeName(value);
  int64_t rank = tensorType.getRank();
  
  // Get static shape if available, otherwise use shape parameter
  bool hasStaticShape = tensorType.hasStaticShape();
  
  // Calculate total size
  oss << "  // Allocate " << varName << "\n";
  oss << "  int64_t " << varName << "_size = 1";
  for (int64_t i = 0; i < rank; i++) {
    if (hasStaticShape && !tensorType.isDynamicDim(i)) {
      oss << " * " << tensorType.getDimSize(i);
    } else {
      // For dynamic dimensions, we need the shape - but we may not have it yet
      // Use shape parameter if available
      oss << " * " << shapeName << "[" << i << "]";
    }
  }
  oss << ";\n";
  
  oss << "  " << elemType << "* " << varName << " = (" << elemType << "*)malloc(" 
      << varName << "_size * sizeof(" << elemType << "));\n";
  
  // If we have static shape, also generate shape array for the allocated tensor
  // This is needed for index calculations
  if (hasStaticShape) {
    oss << "  int64_t " << shapeName << "[] = {";
    for (int64_t i = 0; i < rank; i++) {
      if (i > 0) oss << ", ";
      oss << tensorType.getDimSize(i);
    }
    oss << "};\n";
  } else {
    // For dynamic shapes, we need the shape parameter - but intermediate values
    // don't have shape parameters, so we need to calculate from operands
    // For now, generate a comment indicating this needs to be handled
    oss << "  // TODO: Shape array for " << varName << " (dynamic shape)\n";
  }
  
  return oss.str();
}

std::string CCodeGenerator::generateDeallocation(const std::string& varName) {
  std::ostringstream oss;
  oss << "  free(" << varName << ");";
  return oss.str();
}

void CCodeGenerator::initializeFunctionArgs(func::FuncOp funcOp) {
  Block& entryBlock = funcOp.getBody().front();
  
  // Register argument names
  for (unsigned i = 0; i < entryBlock.getNumArguments(); i++) {
    auto arg = entryBlock.getArgument(i);
    std::string name = "arg" + std::to_string(i);
    registerValue(arg, name);
  }
  
  // Register result names for operations
  for (auto& op : entryBlock) {
    for (unsigned i = 0; i < op.getNumResults(); i++) {
      auto result = op.getResult(i);
      void* valuePtr = result.getAsOpaquePointer();
      if (valueNames.find(valuePtr) == valueNames.end()) {
        std::string name = generateTempName();
        registerValue(result, name);
      }
    }
  }
}

std::string CCodeGenerator::getValueName(Value value) {
  // Use value pointer as key since Value doesn't have comparison operators
  void* valuePtr = value.getAsOpaquePointer();
  auto it = valueNames.find(valuePtr);
  if (it != valueNames.end()) {
    return it->second;
  }
  std::string name = "temp" + std::to_string(tempCounter++);
  registerValue(value, name);
  return name;
}

std::string CCodeGenerator::getShapeName(Value value) {
  void* valuePtr = value.getAsOpaquePointer();
  auto it = shapeNames.find(valuePtr);
  if (it != shapeNames.end()) {
    return it->second;
  }
  
  // Generate shape name from value name
  std::string valueName = getValueName(value);
  std::string shapeName = valueName + "_shape";
  shapeNames[valuePtr] = shapeName;
  
  // If it's a static tensor, we can generate the shape array inline
  // Otherwise, we'll need to use the shape parameter
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.hasStaticShape()) {
      // For static shapes, we can generate the shape inline when needed
      // The shape name will be used in index calculations
    }
  }
  
  return shapeName;
}

std::string CCodeGenerator::generateTempName() {
  return "temp" + std::to_string(tempCounter++);
}

void CCodeGenerator::registerValue(Value value, const std::string& name) {
  void* valuePtr = value.getAsOpaquePointer();
  valueNames[valuePtr] = name;
}

bool CCodeGenerator::isSupported(Operation* op) {
  return COpEmitters::isSupported(op);
}

} // namespace stablehlo
} // namespace mlir
