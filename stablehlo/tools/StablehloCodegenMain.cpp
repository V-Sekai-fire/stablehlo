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
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
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
#include "mlir/Transforms/Passes.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
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

// Lower StableHLO to LLVM IR through a series of passes
LogicalResult lowerToLLVMIR(ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);

  // Step 1: Convert StableHLO to Linalg
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());

  // Step 2: Bufferize operations (convert tensors to memrefs)
  pm.addPass(bufferization::createOneShotBufferizePass());

  // Step 3: Convert Linalg to loops
  pm.addPass(createConvertLinalgToLoopsPass());

  // Step 4: Convert SCF to control flow
  pm.addPass(createSCFToControlFlowPass());

  // Step 5: Lower Affine operations
  pm.addPass(createLowerAffinePass());

  // Step 6: Convert to LLVM dialect
  // Note: Order matters - Arith must come before Func, and MemRef must be finalized last
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());

  // Step 7: Finalize LLVM dialect (handles MemRef conversion)
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  
  // Step 8: Final reconcile to clean up any remaining casts
  pm.addPass(createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module))) {
    module->emitError("Pass pipeline failed - dumping module state");
    // Debug output to see what operations remain
    module->print(llvm::errs());
    return failure();
  }

  // Debug: verify conversion worked by checking for func operations
  bool foundFuncOps = false;
  module->walk([&](Operation *op) {
    if (isa<func::FuncOp>(op) || isa<func::ReturnOp>(op) || isa<func::CallOp>(op)) {
      foundFuncOps = true;
      op->emitWarning() << "Found unconverted func operation after conversion pipeline";
    }
  });
  
  if (foundFuncOps) {
    module->emitError("Func operations were not converted to LLVM dialect - dumping module");
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
