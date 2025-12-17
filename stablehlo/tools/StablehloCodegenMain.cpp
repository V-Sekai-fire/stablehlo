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

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"

using namespace mlir;
using namespace llvm;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                            cl::value_desc("filename"),
                                            cl::init("-"));

static cl::opt<std::string> targetTriple("target-triple",
                                          cl::desc("Target triple"),
                                          cl::init("riscv64-unknown-linux-gnu"));

static cl::opt<std::string> targetCPU("cpu", cl::desc("Target CPU"),
                                       cl::init("generic-rv64"));

static cl::opt<std::string> targetFeatures("features",
                                            cl::desc("Target features"),
                                            cl::init("+m,+a,+f,+d"));

static cl::opt<CodeGenOptLevel> optLevel(
    cl::desc("Optimization level:"),
    cl::values(clEnumValN(CodeGenOptLevel::None, "O0", "No optimizations"),
               clEnumValN(CodeGenOptLevel::Less, "O1", "Few optimizations"),
               clEnumValN(CodeGenOptLevel::Default, "O2", "Default optimizations"),
               clEnumValN(CodeGenOptLevel::Aggressive, "O3",
                          "Aggressive optimizations")),
    cl::init(CodeGenOptLevel::Default));

static cl::opt<bool> emitLLVM("emit-llvm",
                               cl::desc("Emit LLVM IR instead of binary"),
                               cl::init(false));

static cl::opt<bool> emitAsm("emit-asm",
                              cl::desc("Emit assembly instead of binary"),
                              cl::init(false));

// Type converter that converts tensor types to memref types
// This is needed because OneShotBufferizePass doesn't convert function signatures
//
// TOMBSTONE: Failed approach - Initially tried to convert both function inputs AND returns
// from tensor to memref. This caused issues because:
// 1. Return types don't need to match syscalls (only inputs do)
// 2. Converting returns created complex materialization chains with unreconcilable casts
// 3. LLVM translation failed on functions with tensor returns
// SOLUTION: Convert inputs to memref (for syscalls) and convert returns to void
// (discard tensor returns since we only need syscalls, not return values)
class TensorToMemRefTypeConverter : public TypeConverter {
public:
  TensorToMemRefTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](RankedTensorType type) -> mlir::Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    addConversion([](UnrankedTensorType type) -> mlir::Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    
    // Add materialization for memref -> tensor (for function arguments)
    // This allows converting function arguments from memref back to tensor
    // if needed by operations in the function body
    addSourceMaterialization([](OpBuilder &builder, mlir::Type type,
                                ValueRange values, Location loc) -> mlir::Value {
      if (values.size() != 1)
        return {};
      if (auto memrefType = dyn_cast<MemRefType>(values[0].getType())) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
          if (memrefType.getShape() == tensorType.getShape() &&
              memrefType.getElementType() == tensorType.getElementType()) {
            // Use bufferization.to_tensor to convert memref to tensor
            return builder.create<bufferization::ToTensorOp>(
                loc, tensorType, values[0]);
          }
        }
      }
      return {};
    });
    
    // Add materialization for tensor -> memref (for function returns)
    // This allows converting function return values from tensor to memref
    addTargetMaterialization([](OpBuilder &builder, mlir::Type type,
                                ValueRange values, Location loc) -> mlir::Value {
      if (values.size() != 1)
        return {};
      
      mlir::Type valueType = values[0].getType();
      mlir::Type targetType = type;
      
      // Handle ranked tensor to memref conversion
      if (auto tensorType = dyn_cast<RankedTensorType>(valueType)) {
        if (auto memrefType = dyn_cast<MemRefType>(targetType)) {
          // Check if shapes and element types match (ignore layout for now)
          if (memrefType.getShape() == tensorType.getShape() &&
              memrefType.getElementType() == tensorType.getElementType()) {
            // Use bufferization.to_buffer to convert tensor to memref
            // Create with the target memref type - ToBufferOp will handle layout
            auto resultType = MemRefType::get(
                tensorType.getShape(),
                tensorType.getElementType(),
                memrefType.getLayout(),  // Use target layout
                memrefType.getMemorySpace());
            return bufferization::ToBufferOp::create(builder, loc, resultType, values[0]);
          }
        }
      }
      
      // Handle unranked tensor to unranked memref conversion
      if (auto tensorType = dyn_cast<UnrankedTensorType>(valueType)) {
        if (auto memrefType = dyn_cast<UnrankedMemRefType>(targetType)) {
          if (memrefType.getElementType() == tensorType.getElementType()) {
            return bufferization::ToBufferOp::create(builder, loc, memrefType, values[0]);
          }
        }
      }
      
      return {};
    });
  }
};

// Custom pattern to convert function inputs to memref and remove tensor returns
// Since we only need syscalls (inputs), we convert to void return
//
// TOMBSTONE: Failed approach - Initially tried to preserve tensor return types
// and only convert inputs. This failed because:
// 1. createConvertFuncToLLVMPass() requires all function types to be LLVM-convertible
// 2. LLVMTypeConverter cannot handle tensor types
// 3. Functions with tensor returns couldn't be converted to llvm.func
// SOLUTION: Convert returns to void (empty return type) - the computation still happens,
// we just don't return the tensor value. This allows all functions to be converted to LLVM.
struct ConvertFuncForSyscallsPattern : public OpConversionPattern<func::FuncOp> {
  ConvertFuncForSyscallsPattern(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern(converter, context) {}

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcType = op.getFunctionType();
    const TypeConverter *converter = getTypeConverter();
    
    // Convert input types to memref (for syscalls)
    SmallVector<mlir::Type> convertedInputTypes;
    if (failed(converter->convertTypes(funcType.getInputs(), convertedInputTypes)))
      return failure();
    
    // Remove return types - convert to void return since we only need syscalls
    // The computation still happens, we just don't return the tensor
    SmallVector<mlir::Type> voidReturns; // Empty = void return
    
    // Create new function type with converted inputs and void return
    auto newFuncType = rewriter.getFunctionType(convertedInputTypes, voidReturns);
    
    // Create new function with converted signature
    auto newFunc = func::FuncOp::create(
        rewriter, op.getLoc(), op.getName(), newFuncType,
        op.getSymVisibilityAttr(), op.getArgAttrsAttr(), ArrayAttr()); // No result attrs
    
    // Move function body
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
    
    // Convert entry block arguments using signature conversion
    TypeConverter::SignatureConversion signatureConversion(funcType.getNumInputs());
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      signatureConversion.addInputs(i, {convertedInputTypes[i]});
    }
    
    // Apply signature conversion to entry block
    if (failed(rewriter.convertRegionTypes(&newFunc.getBody(), *converter,
                                           &signatureConversion)))
      return failure();
    
    // Replace all return operations with void returns (discard tensor returns)
    newFunc.walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp);
    });
    
    rewriter.replaceOp(op, newFunc);
    return success();
  }
};

// Pass to convert function signatures from tensor to memref types
// This is needed because OneShotBufferizePass doesn't convert function signatures
// We only convert INPUT arguments (for syscalls), not return types
//
// TOMBSTONE: Failed approach - Initially tried using populateFunctionOpInterfaceTypeConversionPattern
// which converts both inputs AND outputs. This caused:
// 1. Return type mismatches (function signature had memref, return had tensor)
// 2. Complex materialization chains that couldn't be reconciled
// SOLUTION: Custom pattern (ConvertFuncForSyscallsPattern) that only converts inputs
// and explicitly converts returns to void
//
// TOMBSTONE: Failed approach - Tried to mark stablehlo::CustomCallOp as legal to preserve
// through the pipeline. This failed because:
// 1. Bufferization couldn't handle custom_call operations (they weren't bufferized)
// 2. LLVM conversion doesn't know how to handle stablehlo.custom_call
// SOLUTION: Convert custom_call to func.call (see Step 2.6) so they become regular
// function calls that are preserved through the pipeline
struct ConvertFunctionSignaturesPass
    : public PassWrapper<ConvertFunctionSignaturesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();
    
    TensorToMemRefTypeConverter converter;
    ConversionTarget target(*context);
    
    // Mark func.func as illegal if it has tensor types in inputs OR returns
    // We convert inputs to memref and remove returns (void return)
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto funcType = op.getFunctionType();
      // Check input types - must be memref (not tensor)
      for (mlir::Type inputType : funcType.getInputs()) {
        if (isa<TensorType>(inputType))
          return false;
      }
      // Check return types - must be void (no tensor returns)
      if (!funcType.getResults().empty())
        return false;
      return true;
    });
    
    // func.return is legal (will be converted to void returns)
    target.addLegalOp<func::ReturnOp>();
    
    // custom_call operations are already converted to func.call in Step 1.5
    // All other operations are legal
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    
    RewritePatternSet patterns(context);
    // Use custom pattern that converts inputs and removes tensor returns (void return)
    patterns.add<ConvertFuncForSyscallsPattern>(converter, context);
    
    // Use applyFullConversion to ensure all operations are converted
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    
    // Cleanup unrealized conversion casts (if any, created during conversion)
    SmallVector<UnrealizedConversionCastOp> casts;
    module->walk([&](UnrealizedConversionCastOp op) { 
      casts.push_back(op); 
    });
    if (!casts.empty()) {
      // Reconcile the casts - this should eliminate them if possible
      mlir::reconcileUnrealizedCasts(casts);
    }
  }
};

// Lower StableHLO to LLVM IR through a series of passes
LogicalResult lowerToLLVMIR(ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);

  // Step 1: Convert StableHLO to Linalg
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());

  // Step 2: Bufferize operations (convert tensors to memrefs)
  // For RISC-V CPU target, bufferization converts tensors to memrefs
  pm.addPass(bufferization::createOneShotBufferizePass());
  
  // TOMBSTONE: Failed approach - Tried to convert stablehlo.custom_call to func.call
  // BEFORE bufferization. This failed because:
  // 1. Function signatures were created with tensor types
  // 2. After bufferization, operands became memref but function signature expected tensor
  // 3. Type mismatches: "expected operand type 'memref<...>', but provided 'tensor<...>'"
  // SOLUTION: Convert custom_call AFTER bufferization and function signature conversion
  // (see Step 2.6 below) so types are already memref
  
  // Step 2.5: Convert function signatures from tensor to memref types
  // OneShotBufferizePass doesn't convert function signatures by default
  // This is needed for CPU codegen where func.func must have memref types
  pm.addPass(std::make_unique<ConvertFunctionSignaturesPass>());
  
  // Step 3: Convert Linalg to loops
  pm.addPass(createConvertLinalgToLoopsPass());

  // Step 4: Convert SCF to control flow
  pm.addPass(createSCFToControlFlowPass());

  // Step 5: Lower Affine operations
  pm.addPass(createLowerAffinePass());

  // Step 6: Finalize MemRef to LLVM (must come before Func conversion)
  // Based on TestLowerToLLVM example: MemRef finalization comes before Func
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  
  // Step 7: Convert to LLVM dialect
  // Order based on TestLowerToLLVM example:
  // 1. Func to LLVM (all functions now have void returns, so all can be converted)
  // 2. Arith to LLVM
  // 3. ControlFlow to LLVM
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  
  // Step 8: Final reconcile to clean up any remaining casts
  // This is critical - all unrealized casts must be reconciled before LLVM translation
  pm.addPass(createReconcileUnrealizedCastsPass());
  
  // Step 9: Additional reconciliation pass to ensure all casts are resolved
  // Sometimes multiple passes are needed for complex cast chains
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    module->emitError("Pass pipeline failed - dumping module state");
    // Debug output to see what operations remain
    module->print(llvm::errs());
    return failure();
  }
  
  // Step 2.7: Convert stablehlo.custom_call to func.call (preserve Godot syscalls)
  // This must happen AFTER all passes run, so types are already memref
  // TOMBSTONE: Failed approach - Tried to preserve custom_call through the pipeline
  // by marking it as legal. This failed because:
  // 1. Bufferization couldn't handle custom_call (op was not bufferized error)
  // 2. LLVM conversion doesn't support stablehlo.custom_call
  // SOLUTION: Convert to func.call which becomes llvm.call naturally
  SymbolTable symbolTable(module);
  SmallVector<stablehlo::CustomCallOp> customCalls;
  module->walk([&](stablehlo::CustomCallOp customCall) {
    customCalls.push_back(customCall);
  });
  
  for (auto customCall : customCalls) {
    OpBuilder builder(customCall);
    builder.setInsertionPoint(customCall);
    
    StringRef callTarget = customCall.getCallTargetName();
    
    // Get operand types (should be memref after bufferization)
    SmallVector<mlir::Type> inputTypes;
    for (auto input : customCall.getInputs()) {
      inputTypes.push_back(input.getType());
    }
    SmallVector<mlir::Type> resultTypes;
    for (auto result : customCall.getResults()) {
      resultTypes.push_back(result.getType());
    }
    auto funcType = builder.getFunctionType(inputTypes, resultTypes);
    
    // Create function declaration if it doesn't exist (external syscall implementation)
    if (!symbolTable.lookup<func::FuncOp>(callTarget)) {
      auto declFunc = func::FuncOp::create(
          builder, customCall.getLoc(), callTarget, funcType);
      declFunc.setPrivate(); // External linkage - provided by syscall implementation
      symbolTable.insert(declFunc);
    }
    
    // Convert custom_call to func.call (will become llvm.call)
    auto callOp = func::CallOp::create(
        builder,
        customCall.getLoc(),
        resultTypes,
        callTarget,
        customCall.getInputs());
    
    customCall.replaceAllUsesWith(callOp.getResults());
    customCall.erase();
  }
  
  // Final reconciliation for any remaining casts
  SmallVector<UnrealizedConversionCastOp> finalCasts;
  module->walk([&](UnrealizedConversionCastOp op) { 
    finalCasts.push_back(op); 
  });
  if (!finalCasts.empty()) {
    mlir::reconcileUnrealizedCasts(finalCasts);
  }

  // Debug: verify conversion worked - all func.func should be converted to llvm.func
  // (All functions now have void returns, so they should all be converted)
  bool foundFuncOps = false;
  module->walk([&](func::FuncOp op) {
    foundFuncOps = true;
    op.emitWarning() << "Found unconverted func.func operation after conversion pipeline";
  });
  
  if (foundFuncOps) {
    module->emitError("func.func operations were not converted to llvm.func - dumping module");
    module->print(llvm::errs());
    return failure();
  }

  return success();
}

// Convert LLVM IR module to RISC-V code
LogicalResult emitRiscVCode(ModuleOp module, StringRef outputPath,
                            StringRef triple, StringRef cpu,
                            StringRef features, CodeGenOptLevel optLvl,
                            bool emitLLVMIR, bool emitAssembly) {
  // TOMBSTONE: Failed approach - Tried to remove functions with tensor returns
  // before translation. This failed because:
  // 1. Lost the computation entirely
  // 2. Functions were needed for the module structure
  // 3. translateModuleToLLVMIR still encountered func.func operations
  // SOLUTION: Convert all functions to have void returns, so no removal needed
  
  // All functions now have void returns (tensor returns removed), so no need to remove anything
  // Convert MLIR module to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return module.emitError("failed to convert to LLVM IR");
  }

  llvm::Triple targetTriple(llvm::Triple::normalize(triple));
  llvmModule->setTargetTriple(targetTriple);
  llvmModule->setDataLayout("");

  // Initialize LLVM targets
  InitializeAllTargetInfos();
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  // Lookup RISC-V target
  std::string error;
  const Target *target =
      TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    return module.emitError("failed to lookup target: " + error);
  }

  // Create target machine
  TargetOptions targetOptions;
  std::unique_ptr<TargetMachine> targetMachine(
      target->createTargetMachine(targetTriple, cpu, features, targetOptions,
                                  std::nullopt, std::nullopt, optLvl));

  if (!targetMachine) {
    return module.emitError("failed to create target machine");
  }

  llvmModule->setDataLayout(targetMachine->createDataLayout());

  // Open output file
  std::error_code ec;
  raw_fd_ostream dest(outputPath, ec, sys::fs::OF_None);
  if (ec) {
    return module.emitError("failed to open output file: " + ec.message());
  }

  // Emit code
  legacy::PassManager pass;
  if (emitLLVMIR) {
    // Emit LLVM IR
    dest << *llvmModule;
  } else if (emitAssembly) {
    // Emit assembly
    if (targetMachine->addPassesToEmitFile(
            pass, dest, nullptr, CodeGenFileType::AssemblyFile)) {
      return module.emitError("target machine cannot emit assembly");
    }
    pass.run(*llvmModule);
  } else {
    // Emit object file
    if (targetMachine->addPassesToEmitFile(
            pass, dest, nullptr, CodeGenFileType::ObjectFile)) {
      return module.emitError("target machine cannot emit object file");
    }
    pass.run(*llvmModule);
  }

  dest.flush();
  return success();
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);

  cl::ParseCommandLineOptions(argc, argv,
                              "StableHLO to RISC-V ELF64 code generator\n");

  // Register dialects
  DialectRegistry registry;
  registerAllDialects(registry);
  stablehlo::registerAllDialects(registry);

  // Create MLIR context
  MLIRContext context(registry);
  // Ensure LLVM dialect is loaded (needed for conversion passes)
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Parse input file
  OwningOpRef<ModuleOp> module;
  if (inputFilename == "-") {
    // Read from stdin
    std::string inputStr;
    std::string line;
    while (std::getline(std::cin, line)) {
      inputStr += line + "\n";
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBuffer(inputStr, "<stdin>"), llvm::SMLoc());
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
  } else {
    module = parseSourceFile<ModuleOp>(inputFilename, &context);
  }

  if (!module) {
    errs() << "Error: Could not parse input file\n";
    return 1;
  }

  // Lower to LLVM IR
  if (failed(lowerToLLVMIR(*module))) {
    errs() << "Error: Failed to lower to LLVM IR\n";
    // Debug: dump the module to see what operations remain
    module->dump();
    return 1;
  }

  // Register LLVM IR translations on the context AFTER conversion
  // (following the pattern from MLIR Toy example)
  registerBuiltinDialectTranslation(context);
  registerLLVMDialectTranslation(context);

  // Emit RISC-V code
  if (failed(emitRiscVCode(*module, outputFilename, targetTriple, targetCPU,
                           targetFeatures, optLevel, emitLLVM, emitAsm))) {
    errs() << "Error: Failed to emit RISC-V code\n";
    return 1;
  }

  return 0;
}
