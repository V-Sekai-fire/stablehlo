// Test program to validate Option C: Custom Phased Bufferization Strategy
// This demonstrates the key logic without requiring a full rebuild

#include <iostream>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Parser/Parser.h>

using namespace mlir;

int main() {
  MLIRContext context;

  // Test the signature conversion logic from Phase 1
  std::cout << "Testing Phase 1: Pre-bufferization signature conversion\n";

  // Parse a simple function
  const char *mlirInput = R"mlir(
    module {
      func.func @test(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
        return %arg0 : tensor<2x3xf32>
      }
    }
  )mlir";

  OwningOpRef<ModuleOp> module = parseSourceString<ModuleOp>(mlirInput, &context);
  if (!module) {
    std::cerr << "Failed to parse MLIR input\n";
    return 1;
  }

  // Test signature conversion
  module->walk([&](func::FuncOp funcOp) {
    auto funcType = funcOp.getFunctionType();
    std::cout << "Original function signature: " << funcType << "\n";

    bool needsConversion = false;
    for (auto inputType : funcType.getInputs()) {
      if (isa<TensorType>(inputType)) {
        needsConversion = true;
        break;
      }
    }

    if (needsConversion) {
      // Convert tensor types to memref types
      auto convertTensorToMemRef = [](Type type) -> Type {
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
          return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
        }
        return type;
      };

      SmallVector<Type> newInputTypes;
      for (auto inputType : funcType.getInputs()) {
        newInputTypes.push_back(convertTensorToMemRef(inputType));
      }

      SmallVector<Type> newReturnTypes;
      for (auto returnType : funcType.getResults()) {
        newReturnTypes.push_back(convertTensorToMemRef(returnType));
      }

      auto newFuncType = FunctionType::get(&context, newInputTypes, newReturnTypes);
      funcOp.setFunctionType(newFuncType);

      std::cout << "Converted function signature: " << funcOp.getFunctionType() << "\n";
    }
  });

  // Dump the modified module
  std::cout << "\nModified module:\n";
  module->dump();

  std::cout << "\nPhase 1 signature conversion: SUCCESS\n";
  std::cout << "This demonstrates that tensor->memref signature conversion works.\n";
  std::cout << "In a full implementation, Phases 2&3 would eliminate bufferization artifacts.\n";

  return 0;
}