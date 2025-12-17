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
#include "mlir/IR/PatternMatch.h"
#include "stablehlo/conversions/linalg/transforms/Passes.h"
#include "stablehlo/dialect/Register.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/tools/CCodegen.h"
#include "stablehlo/tools/COpEmitters.h"
#include "stablehlo/tools/GDScriptSourceParser.h"

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

static cl::opt<bool> emitC("emit-c",
                           cl::desc("Emit C99 code instead of binary"),
                           cl::init(false));

static cl::opt<bool> inputGDScript("gdscript",
                                   cl::desc("Input is GDScript source code"),
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
      auto valueType = values[0].getType();
      auto memrefType = dyn_cast<MemRefType>(valueType);
      if (!memrefType)
        return {};
      auto tensorType = dyn_cast<RankedTensorType>(type);
      if (!tensorType)
        return {};
      if (memrefType.getShape() == tensorType.getShape() &&
          memrefType.getElementType() == tensorType.getElementType()) {
        // Use bufferization.to_tensor to convert memref to tensor
        return bufferization::ToTensorOp::create(builder, loc, tensorType, values[0]);
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

// Custom pattern to convert function inputs and returns from tensor to memref
// Memref types are LLVM-convertible (become pointers), unlike tensor types
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
    
    // Convert return types to memref (memref is LLVM-convertible, tensor is not)
    SmallVector<mlir::Type> convertedReturnTypes;
    if (failed(converter->convertTypes(funcType.getResults(), convertedReturnTypes)))
      return failure();
    
    // Create new function type with converted inputs and returns
    auto newFuncType = rewriter.getFunctionType(convertedInputTypes, convertedReturnTypes);
    
    // Create new function with converted signature
    auto newFunc = func::FuncOp::create(
        rewriter, op.getLoc(), op.getName(), newFuncType,
        op.getSymVisibilityAttr(), op.getArgAttrsAttr(), op.getResAttrsAttr());
    
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
    
    // Convert return operations - convert tensor return values to memref
    // After bufferization, return values might be tensors (from to_tensor ops),
    // but the function signature expects memrefs
    // We need to eliminate to_tensor operations and use the underlying memref directly
    newFunc.walk([&](func::ReturnOp returnOp) {
      rewriter.setInsertionPoint(returnOp);
      SmallVector<mlir::Value> convertedReturns;
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        auto operand = returnOp.getOperand(i);
        auto expectedType = convertedReturnTypes[i];
        
        // If operand is a tensor from to_tensor, get the underlying memref
        // ToTensorOp takes a memref and returns a tensor, so we get operand 0
        if (auto toTensorOp = operand.getDefiningOp<bufferization::ToTensorOp>()) {
          // Use the memref directly instead of converting tensor back to memref
          operand = toTensorOp->getOperand(0);
        }
        
        // If the operand type doesn't match the expected type, we need to handle it
        // After memref finalization, operands may be LLVM structs but expected type is memref
        // In that case, we should use the operand directly (memref finalization will handle it)
        // Otherwise, materialize conversion
        if (operand.getType() != expectedType) {
          // If operand is already an LLVM struct but expected is memref, this will be handled
          // by memref finalization - just use the operand directly
          // The materialization will create a cast, but that's handled by finalization
          auto materialized = converter->materializeTargetConversion(
              rewriter, returnOp.getLoc(), expectedType, operand);
          if (materialized) {
            convertedReturns.push_back(materialized);
          } else {
            // If materialization fails, use the operand as-is
            // This can happen after memref finalization when types are LLVM structs
            convertedReturns.push_back(operand);
          }
        } else {
          convertedReturns.push_back(operand);
        }
      }
      rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp, convertedReturns);
    });
    
    rewriter.replaceOp(op, newFunc);
    return success();
  }
};

// Custom pass to convert function signatures from tensor to memref types
// This directly modifies function signatures without using the conversion framework
// After bufferization, function bodies use memrefs but signatures still have tensors
// This pass fixes that mismatch by directly converting signatures
struct ConvertFunctionSignaturesPass
    : public PassWrapper<ConvertFunctionSignaturesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *context = &getContext();

    // Helper to convert tensor type to memref type
    auto convertTensorToMemRef = [](mlir::Type type) -> mlir::Type {
      if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      }
      if (auto tensorType = dyn_cast<UnrankedTensorType>(type)) {
        return UnrankedMemRefType::get(tensorType.getElementType(), 0);
      }
      return type; // Not a tensor, return as-is
    };

    // Use SymbolTable for proper function replacement
    SymbolTable symbolTable(module);

    // Walk all functions and convert their signatures
    SmallVector<func::FuncOp> funcs;
    module->walk([&](func::FuncOp funcOp) {
      funcs.push_back(funcOp);
    });

    llvm::errs() << "ConvertFunctionSignaturesPass: Found " << funcs.size() << " functions\n";
    
    for (auto funcOp : funcs) {
      auto funcType = funcOp.getFunctionType();
      bool needsConversion = false;

      // Check if conversion is needed
      for (auto inputType : funcType.getInputs()) {
        if (isa<TensorType>(inputType)) {
          needsConversion = true;
          break;
        }
      }
      if (!needsConversion) {
        for (auto returnType : funcType.getResults()) {
          if (isa<TensorType>(returnType)) {
            needsConversion = true;
            break;
          }
        }
      }

      llvm::errs() << "ConvertFunctionSignaturesPass: Function " << funcOp.getName()
                   << " - needs conversion: " << (needsConversion ? "YES" : "NO") << "\n";
      llvm::errs() << "  Original signature: " << funcType << "\n";

      if (!needsConversion) {
        continue; // Already converted
      }
      
      // Convert input types
      SmallVector<mlir::Type> newInputTypes;
      for (auto inputType : funcType.getInputs()) {
        newInputTypes.push_back(convertTensorToMemRef(inputType));
      }
      
      // Convert return types
      SmallVector<mlir::Type> newReturnTypes;
      for (auto returnType : funcType.getResults()) {
        newReturnTypes.push_back(convertTensorToMemRef(returnType));
      }
      
      // CRITICAL: Update entry block arguments FIRST to memref types
      // This must happen before updating the function signature
      Block *entryBlock = &funcOp.getBody().front();
      for (unsigned i = 0; i < newInputTypes.size() && i < entryBlock->getNumArguments(); ++i) {
        auto arg = entryBlock->getArgument(i);
        auto oldType = arg.getType();
        auto newType = newInputTypes[i];
        
        if (oldType != newType) {
          // Update argument type to memref
          // After bufferization, body operations use memrefs, so arguments should be memrefs too
          arg.setType(newType);
        }
      }
      
      // Get return types from return operations (after processing them)
      SmallVector<mlir::Type> actualReturnTypes;
      funcOp.walk([&](func::ReturnOp returnOp) {
        for (auto operand : returnOp.getOperands()) {
          actualReturnTypes.push_back(operand.getType());
        }
      });
      
      // If no returns found, use the converted return types
      if (actualReturnTypes.empty()) {
        actualReturnTypes = newReturnTypes;
      }
      
      // Create new function type based on actual argument and return types
      auto newFuncType = mlir::FunctionType::get(context, newInputTypes, actualReturnTypes);
      
      // Update function signature using setFunctionType
      funcOp.setFunctionType(newFuncType);

      llvm::errs() << "  New signature: " << funcOp.getFunctionType() << "\n";

      // Update return operations - after bufferization, return values should already be memrefs
      funcOp.walk([&](func::ReturnOp returnOp) {
        SmallVector<mlir::Value> newOperands;
        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
          auto operand = returnOp.getOperand(i);
          
          // If operand is from to_tensor, get the underlying memref
          if (auto toTensorOp = operand.getDefiningOp<bufferization::ToTensorOp>()) {
            operand = toTensorOp->getOperand(0);
          }
          
          newOperands.push_back(operand);
        }
        
        // Only update if operands changed
        if (newOperands.size() == returnOp.getNumOperands()) {
          bool changed = false;
          for (unsigned i = 0; i < newOperands.size(); ++i) {
            if (newOperands[i] != returnOp.getOperand(i)) {
              changed = true;
              break;
            }
          }
          if (changed) {
            OpBuilder returnBuilder(returnOp);
            returnBuilder.create<func::ReturnOp>(returnOp.getLoc(), newOperands);
            returnOp.erase();
          }
        }
      });
    }

    llvm::errs() << "ConvertFunctionSignaturesPass: Completed\n";
  }
};

// Lower StableHLO to LLVM IR through a series of passes
LogicalResult lowerToLLVMIR(ModuleOp module) {
  llvm::errs() << "Starting StableHLO to LLVM lowering pipeline\n";
  mlir::PassManager pm(module.getContext());
  pm.enableVerifier(true);

  // Step 1: Convert StableHLO to Linalg
  pm.addPass(mlir::stablehlo::createStablehloLegalizeToLinalgPass());

  // Step 1.5: Convert function signatures from tensor to memref types BEFORE bufferization
  // This prevents bufferization from failing due to type mismatches between function
  // signatures (tensors) and function bodies (memrefs after linalg conversion)
  pm.addPass(std::make_unique<ConvertFunctionSignaturesPass>());

  // Step 2: Custom Phased Bufferization Strategy
  // OPTION C: Custom bufferization that preserves function signature consistency
  // This approach runs bufferization in phases with signature updates in between
  {
    struct PhasedBufferizePass : public PassWrapper<PhasedBufferizePass, OperationPass<ModuleOp>> {
      void runOnOperation() override {
        auto module = getOperation();
        MLIRContext *context = &getContext();

        // Phase 1: Analyze function signatures and prepare for bufferization
        // Convert any remaining tensor signatures to memref before bufferization
        {
          // Helper to convert tensor type to memref type
          auto convertTensorToMemRef = [](mlir::Type type) -> mlir::Type {
            if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
              return MemRefType::get(tensorType.getShape(), tensorType.getElementType());
            }
            if (auto tensorType = dyn_cast<UnrankedTensorType>(type)) {
              return UnrankedMemRefType::get(tensorType.getElementType(), 0);
            }
            return type; // Not a tensor, return as-is
          };

          // Pre-bufferization signature conversion
          module.walk([&](func::FuncOp funcOp) {
            auto funcType = funcOp.getFunctionType();
            bool needsConversion = false;

            // Check if conversion is needed
            for (auto inputType : funcType.getInputs()) {
              if (isa<TensorType>(inputType)) {
                needsConversion = true;
                break;
              }
            }
            if (!needsConversion) {
              for (auto returnType : funcType.getResults()) {
                if (isa<TensorType>(returnType)) {
                  needsConversion = true;
                  break;
                }
              }
            }

            if (needsConversion) {
              // Convert input types
              SmallVector<mlir::Type> newInputTypes;
              for (auto inputType : funcType.getInputs()) {
                newInputTypes.push_back(convertTensorToMemRef(inputType));
              }

              // Convert return types
              SmallVector<mlir::Type> newReturnTypes;
              for (auto returnType : funcType.getResults()) {
                newReturnTypes.push_back(convertTensorToMemRef(returnType));
              }

              // Update entry block arguments to match new input types
              Block *entryBlock = &funcOp.getBody().front();
              for (unsigned i = 0; i < newInputTypes.size() && i < entryBlock->getNumArguments(); ++i) {
                auto arg = entryBlock->getArgument(i);
                auto newType = newInputTypes[i];
                if (arg.getType() != newType) {
                  arg.setType(newType);
                }
              }

              // Update function signature
              auto newFuncType = mlir::FunctionType::get(context, newInputTypes, newReturnTypes);
              funcOp.setFunctionType(newFuncType);
            }
          });
        }

        // Phase 2: Run standard bufferization
        // Function signatures are already converted to memref, so bufferization should work smoothly
        {
          mlir::PassManager bufferizePM(context);
          bufferizePM.addPass(bufferization::createOneShotBufferizePass());

          if (failed(bufferizePM.run(module))) {
            signalPassFailure();
            return;
          }
        }

        // Phase 3: Post-bufferization cleanup and signature reconciliation
        // Eliminate bufferization artifacts and ensure signature consistency
        {
          // Pattern to eliminate to_tensor operations
          struct EliminateToTensorPattern : public OpRewritePattern<bufferization::ToTensorOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                        PatternRewriter &rewriter) const override {
              // Replace to_tensor with its memref operand
              rewriter.replaceOp(op, op->getOperand(0));
              return success();
            }
          };

          // Pattern to eliminate to_buffer operations
          struct EliminateToBufferPattern : public OpRewritePattern<bufferization::ToBufferOp> {
            using OpRewritePattern::OpRewritePattern;
            LogicalResult matchAndRewrite(bufferization::ToBufferOp op,
                                        PatternRewriter &rewriter) const override {
              // Replace to_buffer with its tensor operand
              auto tensorOperand = op.getTensor();
              if (isa<TensorType>(tensorOperand.getType())) {
                rewriter.replaceOp(op, tensorOperand);
                return success();
              }
              return failure();
            }
          };

          // Run cleanup patterns
          GreedyRewriteConfig config;

          for (int iteration = 0; iteration < 3; ++iteration) {
            RewritePatternSet cleanupPatterns(context);
            cleanupPatterns.add<EliminateToTensorPattern>(context, /*benefit=*/10);
            cleanupPatterns.add<EliminateToBufferPattern>(context, /*benefit=*/5);
            
            auto frozenPatterns = mlir::FrozenRewritePatternSet(std::move(cleanupPatterns));
            if (failed(applyPatternsGreedily(module, frozenPatterns, config))) {
              break; // Stop if patterns fail
            }

            // Check if any bufferization ops remain
            bool foundAny = false;
            module.walk([&](bufferization::ToTensorOp op) { foundAny = true; });
            module.walk([&](bufferization::ToBufferOp op) { foundAny = true; });
            if (!foundAny) break;
          }

          // Final signature reconciliation
          module.walk([&](func::FuncOp funcOp) {
            auto funcType = funcOp.getFunctionType();

            // Check if return types need updating based on actual return operations
            SmallVector<mlir::Type> actualReturnTypes;
            funcOp.walk([&](func::ReturnOp returnOp) {
              for (auto operand : returnOp.getOperands()) {
                actualReturnTypes.push_back(operand.getType());
              }
            });

            if (!actualReturnTypes.empty() && actualReturnTypes != funcType.getResults()) {
              // Update function signature to match actual return types
              auto newFuncType = mlir::FunctionType::get(context, funcType.getInputs(), actualReturnTypes);
              funcOp.setFunctionType(newFuncType);
            }
          });
        }
      }
    };

    pm.addPass(std::make_unique<PhasedBufferizePass>());
  }
  
  // TOMBSTONE: Failed approach - Tried to convert stablehlo.custom_call to func.call
  // BEFORE bufferization. This failed because:
  // 1. Function signatures were created with tensor types
  // 2. After bufferization, operands became memref but function signature expected tensor
  // 3. Type mismatches: "expected operand type 'memref<...>', but provided 'tensor<...>'"
  // SOLUTION: Convert custom_call AFTER bufferization and function signature conversion
  // (see Step 2.6 below) so types are already memref
  
  // Step 3: Convert Linalg to loops
  pm.addPass(createConvertLinalgToLoopsPass());

  // Step 4: Convert SCF to control flow
  pm.addPass(createSCFToControlFlowPass());

  // Step 5: Lower Affine operations
  pm.addPass(createLowerAffinePass());
  
  // Step 5.5: Eliminate ALL bufferization operations (to_tensor, to_buffer) before memref finalization
  // CRITICAL ORDER: This MUST happen BEFORE Step 6 (memref finalization)
  // After memref finalization, memrefs become LLVM structs, but to_tensor/to_buffer
  // still expect memrefs, creating unrealized conversion casts that can't be reconciled.
  // Solution: Eliminate ALL to_tensor/to_buffer operations before finalization
  // Order of elimination: to_buffer first (identity chains), then standalone to_tensor
  {
    // Pattern 1: Eliminate to_buffer that takes input from to_tensor (identity chain)
    struct EliminateToBufferPattern : public OpRewritePattern<bufferization::ToBufferOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(bufferization::ToBufferOp op,
                                     PatternRewriter &rewriter) const override {
        // If the input is from to_tensor, we can eliminate both
        if (auto toTensor = op.getTensor().getDefiningOp<bufferization::ToTensorOp>()) {
          // Replace to_buffer with the original memref from to_tensor
          rewriter.replaceOp(op, toTensor->getOperand(0));
          return success();
        }
        return failure();
      }
    };
    
    // Pattern 2: Eliminate ALL to_tensor operations (replace with underlying memref)
    // This handles all cases: direct memref, unrealized casts, or any other case
    // CRITICAL: We must eliminate ALL to_tensor operations before memref finalization
    struct EliminateToTensorPattern : public OpRewritePattern<bufferization::ToTensorOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                      PatternRewriter &rewriter) const override {
        auto operand = op->getOperand(0);
        
        // Always replace to_tensor with its operand, regardless of type
        // This eliminates the tensor intermediate - the operand (memref or cast) will be handled by later passes
        rewriter.replaceOp(op, operand);
        return success();
      }
    };
    
    RewritePatternSet patterns(module.getContext());
    // Add to_buffer pattern first (higher priority - eliminates identity chains)
    patterns.add<EliminateToBufferPattern>(module.getContext(), /*benefit=*/10);
    // Add to_tensor pattern (lower priority - handles standalone cases)
    patterns.add<EliminateToTensorPattern>(module.getContext(), /*benefit=*/5);
    
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      return module.emitError("Failed to eliminate bufferization operations");
    }
  }

  // Step 6: Finalize MemRef to LLVM (must come before Func conversion)
  // Custom pass that finalizes memrefs AND converts function signatures
  // The standard pass doesn't convert function return types, causing materialization casts
  {
    struct FinalizeMemRefToLLVMPass
        : public PassWrapper<FinalizeMemRefToLLVMPass, OperationPass<ModuleOp>> {
      void runOnOperation() override {
        auto module = getOperation();
        MLIRContext *context = &getContext();
        
        // First, run the standard memref finalization pass
        mlir::PassManager nestedPM(context);
        nestedPM.addPass(createFinalizeMemRefToLLVMConversionPass());
        if (failed(nestedPM.run(module))) {
          signalPassFailure();
          return;
        }
        
        // Then, manually convert function return types from memref to LLVM struct
        // This is needed because the standard pass doesn't convert function signatures
        mlir::LLVMTypeConverter typeConverter(context);
        module->walk([&](func::FuncOp funcOp) {
          auto funcType = funcOp.getFunctionType();
          SmallVector<mlir::Type> newInputTypes;
          SmallVector<mlir::Type> newReturnTypes;
          
          // Convert input types (should already be LLVM types after finalization)
          for (auto inputType : funcType.getInputs()) {
            newInputTypes.push_back(inputType);
          }
          
          // Convert return types from memref to LLVM struct
          for (auto returnType : funcType.getResults()) {
            if (auto memrefType = dyn_cast<MemRefType>(returnType)) {
              // Convert memref to LLVM struct using LLVMTypeConverter
              auto convertedType = typeConverter.convertType(memrefType);
              if (convertedType) {
                newReturnTypes.push_back(convertedType);
              } else {
                // If conversion fails, keep the memref type (shouldn't happen)
                newReturnTypes.push_back(returnType);
              }
            } else {
              // Not a memref, keep as-is
              newReturnTypes.push_back(returnType);
            }
          }
          
          // Update function signature if return types changed
          if (newReturnTypes != funcType.getResults()) {
            auto newFuncType = mlir::FunctionType::get(context, newInputTypes, newReturnTypes);
            funcOp.setType(newFuncType);
            
            // Also update return operations to convert memref values to LLVM structs
            // After finalization, return values should be LLVM structs, but if they're still
            // memrefs, we need to convert them using the type converter
            funcOp.walk([&](func::ReturnOp returnOp) {
              SmallVector<mlir::Value> newReturns;
              OpBuilder builder(returnOp);
              for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
                auto operand = returnOp.getOperand(i);
                auto expectedType = newReturnTypes[i];
                
                if (operand.getType() != expectedType) {
                  // The operand is still a memref, but we need an LLVM struct
                  // Use the type converter to convert it
                  auto convertedType = typeConverter.convertType(operand.getType());
                  if (convertedType == expectedType) {
                    // Type converter can convert this type - use materialization
                    auto materialized = typeConverter.materializeTargetConversion(
                        builder, returnOp.getLoc(), expectedType, operand);
                    if (materialized) {
                      newReturns.push_back(materialized);
                    } else {
                      // If materialization fails, add a cast (shouldn't happen)
                      auto cast = UnrealizedConversionCastOp::create(builder,
                          returnOp.getLoc(), expectedType, operand);
                      newReturns.push_back(cast.getResult(0));
                    }
                  } else {
                    // Type converter can't convert - add a cast
                    auto cast = UnrealizedConversionCastOp::create(builder,
                        returnOp.getLoc(), expectedType, operand);
                    newReturns.push_back(cast.getResult(0));
                  }
                } else {
                  newReturns.push_back(operand);
                }
              }
              func::ReturnOp::create(builder, returnOp.getLoc(), newReturns);
              returnOp.erase();
            });
          }
        });
      }
    };
    
    pm.addPass(std::make_unique<FinalizeMemRefToLLVMPass>());
  }
  
  // Step 6.5: Eliminate any remaining bufferization operations AFTER memref finalization
  // Materialization functions may create to_tensor operations after finalization
  // These must be eliminated before Func conversion
  {
    struct EliminateAllToTensorPattern : public OpRewritePattern<bufferization::ToTensorOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(bufferization::ToTensorOp op,
                                      PatternRewriter &rewriter) const override {
        // Always replace to_tensor with its operand - eliminate all to_tensor operations
        rewriter.replaceOp(op, op->getOperand(0));
        return success();
      }
    };
    
    struct EliminateAllToBufferPattern : public OpRewritePattern<bufferization::ToBufferOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(bufferization::ToBufferOp op,
                                     PatternRewriter &rewriter) const override {
        // If input is from to_tensor, eliminate both; otherwise just eliminate to_buffer
        if (auto toTensor = op.getTensor().getDefiningOp<bufferization::ToTensorOp>()) {
          rewriter.replaceOp(op, toTensor->getOperand(0));
        } else {
          // Can't eliminate to_buffer without to_tensor - this shouldn't happen after finalization
          return failure();
        }
        return success();
      }
    };
    
    RewritePatternSet patterns(module.getContext());
    patterns.add<EliminateAllToTensorPattern>(module.getContext(), /*benefit=*/10);
    patterns.add<EliminateAllToBufferPattern>(module.getContext(), /*benefit=*/5);
    
    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      return module.emitError("Failed to eliminate bufferization operations after memref finalization");
    }
  }
  
  // Step 7: Convert to LLVM dialect
  // Order based on TestLowerToLLVM example:
  // 1. Func to LLVM (all functions now have memref returns, which are LLVM-convertible)
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
    llvm::errs() << "Found " << finalCasts.size() << " unrealized conversion casts before reconciliation\n";
    for (auto cast : finalCasts) {
      llvm::errs() << "  Cast: ";
      cast.print(llvm::errs());
      llvm::errs() << "\n";
    }
    mlir::reconcileUnrealizedCasts(finalCasts);
    
    // Check again after reconciliation
    SmallVector<UnrealizedConversionCastOp> remainingCasts;
    module->walk([&](UnrealizedConversionCastOp op) { 
      remainingCasts.push_back(op); 
    });
    if (!remainingCasts.empty()) {
      llvm::errs() << "WARNING: " << remainingCasts.size() << " unrealized conversion casts remain after reconciliation\n";
      for (auto cast : remainingCasts) {
        llvm::errs() << "  Remaining cast: ";
        cast.print(llvm::errs());
        llvm::errs() << "\n";
        cast.emitError() << "Unreconciled conversion cast";
      }
      module->print(llvm::errs());
      return failure();
    }
  }

  // Debug: verify conversion worked - all func.func should be converted to llvm.func
  // (All functions now have memref returns, which are LLVM-convertible)
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
  // SOLUTION: Convert all functions to have memref returns (LLVM-convertible), so no removal needed
  
  // All functions now have memref returns (tensor returns converted to memref), so no need to remove anything
  
  // Final check for unrealized conversion casts before LLVM translation
  SmallVector<UnrealizedConversionCastOp> preTranslationCasts;
  module.walk([&](UnrealizedConversionCastOp op) { 
    preTranslationCasts.push_back(op); 
  });
  if (!preTranslationCasts.empty()) {
    llvm::errs() << "ERROR: Found " << preTranslationCasts.size() << " unrealized conversion casts before LLVM translation:\n";
    for (auto cast : preTranslationCasts) {
      llvm::errs() << "  Cast at ";
      cast.getLoc().print(llvm::errs());
      llvm::errs() << ": ";
      cast.print(llvm::errs());
      llvm::errs() << "\n";
      cast.emitError() << "Unrealized conversion cast that cannot be translated to LLVM";
    }
    module.print(llvm::errs());
    return module.emitError("unrealized conversion casts remain - cannot translate to LLVM IR");
  }
  
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

// Generate C99 code from StableHLO module
LogicalResult generateCCode(ModuleOp module, StringRef outputFile) {
  stablehlo::CCodeGenerator generator;
  
  // Check for unsupported operations before generating
  bool hasUnsupported = false;
  module.walk([&](Operation* op) {
    if (!stablehlo::COpEmitters::isSupported(op) && 
        !op->hasTrait<OpTrait::IsTerminator>() &&
        !isa<func::FuncOp>(op) &&
        !isa<ModuleOp>(op)) {
      hasUnsupported = true;
      op->emitError() << "Operation " << op->getName() 
                      << " is not supported by C codegen backend";
    }
  });
  
  if (hasUnsupported) {
    return failure();
  }
  
  std::string cCode = generator.generateModule(module);
  
  std::error_code ec;
  raw_fd_ostream os(outputFile, ec);
  if (ec) {
    return module.emitError("Failed to open output file: " + ec.message());
  }
  
  os << cCode;
  os.flush();
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

  llvm::errs() << "Parsing input file: " << inputFilename << "\n";
  
  // Check if input is GDScript (by flag or file extension)
  bool isGDScript = inputGDScript;
  if (!isGDScript && inputFilename != "-") {
    std::string filename = inputFilename;
    if (filename.length() >= 3 && 
        filename.substr(filename.length() - 3) == ".gd") {
      isGDScript = true;
    }
  }
  
  // Parse input file
  OwningOpRef<ModuleOp> module;
  if (isGDScript) {
    // Parse GDScript source code
    std::string sourceCode;
    if (inputFilename == "-") {
      // Read from stdin
      std::string line;
      while (std::getline(std::cin, line)) {
        sourceCode += line + "\n";
      }
    } else {
      // Read from file
      auto bufferOrError = llvm::MemoryBuffer::getFile(inputFilename);
      if (auto ec = bufferOrError.getError()) {
        errs() << "Error: Could not read GDScript file: " << ec.message() << "\n";
        return 1;
      }
      auto buffer = std::move(*bufferOrError);
      sourceCode = buffer->getBuffer().str();
    }
    
    llvm::errs() << "Parsing GDScript source code...\n";
    if (failed(stablehlo::GDScriptSourceParser::parseToStableHLO(
            sourceCode, inputFilename, &context, module))) {
      errs() << "Error: Failed to parse GDScript source code\n";
      return 1;
    }
    
    if (!module) {
      errs() << "Error: GDScript parsing produced no module\n";
      return 1;
    }
    
    llvm::errs() << "Successfully parsed GDScript source code\n";
  } else {
    // Parse MLIR file (existing path)
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

    llvm::errs() << "Successfully parsed input file\n";
  }

  // Check if we should emit C code
  if (emitC) {
    if (failed(generateCCode(*module, outputFilename))) {
      errs() << "Error: Failed to generate C code\n";
      return 1;
    }
    llvm::errs() << "Successfully generated C99 code to " << outputFilename << "\n";
    return 0;
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
