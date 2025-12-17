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

#include "stablehlo/tools/CTypeConverter.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace stablehlo {

std::string CTypeConverter::convertElementType(Type elemType) {
  if (elemType.isF32()) {
    return "float";
  } else if (elemType.isF64()) {
    return "double";
  } else if (elemType.isInteger(32)) {
    return "int32_t";
  } else if (elemType.isInteger(64)) {
    return "int64_t";
  } else if (elemType.isInteger(1)) {
    return "bool";
  } else if (elemType.isInteger(8)) {
    return "int8_t";
  } else if (elemType.isInteger(16)) {
    return "int16_t";
  } else if (elemType.isUnsignedInteger(8)) {
    return "uint8_t";
  } else if (elemType.isUnsignedInteger(16)) {
    return "uint16_t";
  } else if (elemType.isUnsignedInteger(32)) {
    return "uint32_t";
  } else if (elemType.isUnsignedInteger(64)) {
    return "uint64_t";
  }
  
  // Default: try to get as integer
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    unsigned width = intType.getWidth();
    if (intType.isUnsigned()) {
      if (width <= 8) return "uint8_t";
      if (width <= 16) return "uint16_t";
      if (width <= 32) return "uint32_t";
      return "uint64_t";
    } else {
      if (width <= 8) return "int8_t";
      if (width <= 16) return "int16_t";
      if (width <= 32) return "int32_t";
      return "int64_t";
    }
  }
  
  return "void"; // Unsupported type
}

std::string CTypeConverter::convertType(Type type, bool useRestrict) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    auto elemType = tensorType.getElementType();
    std::string cType = convertElementType(elemType);
    if (useRestrict) {
      cType += "* restrict";
    } else {
      cType += "*";
    }
    return cType;
  } else if (auto tensorType = dyn_cast<UnrankedTensorType>(type)) {
    auto elemType = tensorType.getElementType();
    std::string cType = convertElementType(elemType);
    cType += "*";
    if (useRestrict) {
      cType = cType.insert(cType.length() - 1, " restrict");
    }
    return cType;
  } else if (type.isIndex()) {
    return "int64_t";
  } else if (type.isInteger(32)) {
    return "int32_t";
  } else if (type.isInteger(64)) {
    return "int64_t";
  } else if (type.isF32()) {
    return "float";
  } else if (type.isF64()) {
    return "double";
  }
  
  return "void";
}

std::string CTypeConverter::getShapeParamName(StringRef tensorName, int64_t rank) {
  return (tensorName + "_shape").str();
}

bool CTypeConverter::isSupportedType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return isSupportedType(tensorType.getElementType());
  } else if (auto tensorType = dyn_cast<UnrankedTensorType>(type)) {
    return isSupportedType(tensorType.getElementType());
  } else if (type.isF32() || type.isF64()) {
    return true;
  } else if (type.isIndex() || type.isInteger()) {
    return true;
  }
  return false;
}

} // namespace stablehlo
} // namespace mlir
