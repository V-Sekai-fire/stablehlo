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

#include "stablehlo/tools/GDScriptSourceParser.h"
#include "stablehlo/tools/GDScriptASTToStableHLO.h"
#include "stablehlo/tools/gdscript_parser/gdscript_parser.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace stablehlo {

bool GDScriptSourceParser::isAvailable() {
  // GDScript parser is now available (copied from Godot)
  return true;
}

LogicalResult GDScriptSourceParser::parseToStableHLO(
    llvm::StringRef sourceCode,
    llvm::StringRef sourcePath,
    MLIRContext* context,
    OwningOpRef<ModuleOp>& module) {
  
  // Create GDScriptParser instance
  GDScriptParser parser;
  
  // Parse the source code
  Error err = parser.parse(sourceCode.str(), sourcePath.str(), false);
  if (err != OK) {
    llvm::errs() << "Error: Failed to parse GDScript source code\n";
    return failure();
  }
  
  // Get the AST
  const GDScriptParser::ClassNode* ast = parser.get_tree();
  if (!ast) {
    llvm::errs() << "Error: GDScript parser produced no AST\n";
    return failure();
  }
  
  // Create MLIR module
  module = ModuleOp::create(UnknownLoc::get(context));
  
  // Convert AST to StableHLO
  if (failed(GDScriptASTToStableHLO::convertClassToStableHLO(
          ast, context, *module))) {
    llvm::errs() << "Error: Failed to convert GDScript AST to StableHLO\n";
    return failure();
  }
  
  return success();
}

} // namespace stablehlo
} // namespace mlir
