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

#ifndef STABLEHLO_TOOLS_GDSCRIPTASTTOSTABLEHLO_H_
#define STABLEHLO_TOOLS_GDSCRIPTASTTOSTABLEHLO_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "stablehlo/tools/GDScriptSyscallMap.h"
#include "stablehlo/tools/gdscript_parser/gdscript_parser.h"
#include <string>
#include <unordered_map>

namespace mlir {
namespace stablehlo {

/// Converts GDScriptParser AST nodes to StableHLO MLIR operations
class GDScriptASTToStableHLO {
public:
  /// Convert a GDScript class to StableHLO module
  static LogicalResult convertClassToStableHLO(
      const GDScriptParser::ClassNode* ast,
      MLIRContext* context,
      ModuleOp module);
  
  /// Convert a function node to func.func
  static LogicalResult convertFunctionToStableHLO(
      const GDScriptParser::FunctionNode* funcNode,
      OpBuilder& builder,
      ModuleOp module,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert an expression to StableHLO operation
  static Value convertExpression(
      const GDScriptParser::ExpressionNode* expr,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert binary operation
  static Value convertBinaryOp(
      const GDScriptParser::BinaryOpNode* binOp,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert unary operation
  static Value convertUnaryOp(
      const GDScriptParser::UnaryOpNode* unaryOp,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert function call
  static Value convertCall(
      const GDScriptParser::CallNode* call,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert identifier (variable reference)
  static Value convertIdentifier(
      const GDScriptParser::IdentifierNode* ident,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert literal value
  static Value convertLiteral(
      const GDScriptParser::LiteralNode* literal,
      OpBuilder& builder,
      Location loc);
  
  /// Convert suite (block of statements)
  static LogicalResult convertSuite(
      const GDScriptParser::SuiteNode* suite,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert if statement
  static LogicalResult convertIf(
      const GDScriptParser::IfNode* ifNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert while loop
  static LogicalResult convertWhile(
      const GDScriptParser::WhileNode* whileNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert for loop
  static LogicalResult convertFor(
      const GDScriptParser::ForNode* forNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert variable declaration
  static Value convertVariable(
      const GDScriptParser::VariableNode* varNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap);
  
  /// Convert assignment
  static LogicalResult convertAssignment(
      const GDScriptParser::AssignmentNode* assignNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert return statement
  static LogicalResult convertReturn(
      const GDScriptParser::ReturnNode* returnNode,
      OpBuilder& builder,
      Location loc,
      std::unordered_map<std::string, Value>& variableMap,
      Block* block);
  
  /// Convert GDScript type to MLIR type
  static Type convertGDScriptTypeToMLIR(
      const std::string& gdscriptType,
      MLIRContext* context);
  
  /// Get MLIR type for a value (infer from expression if needed)
  static Type inferType(
      const GDScriptParser::ExpressionNode* expr,
      MLIRContext* context);

private:
  /// Helper to create tensor type
  static RankedTensorType createTensorType(
      Type elementType,
      MLIRContext* context);
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_GDSCRIPTASTTOSTABLEHLO_H_
