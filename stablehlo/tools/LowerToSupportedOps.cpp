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

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::stablehlo;

namespace {

// Lower stablehlo.minimum to stablehlo.select + stablehlo.compare
// minimum(a, b) = select(a < b, a, b)
struct LowerMinimumOp : public OpRewritePattern<MinOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MinOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // Determine comparison type
    ComparisonType compareType = resultType.getElementType().isUnsignedInteger() 
        ? ComparisonType::UNSIGNED : ComparisonType::SIGNED;
    auto compareTypeAttr = ComparisonTypeAttr::get(rewriter.getContext(), compareType);
    
    // Create comparison: lhs < rhs
    auto compare = CompareOp::create(
        rewriter, loc,
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type()),
        lhs, rhs,
        ComparisonDirection::LT,
        compareTypeAttr);
    
    // Select: if lhs < rhs then lhs else rhs
    auto select = SelectOp::create(rewriter, loc, resultType, compare, lhs, rhs);
    
    rewriter.replaceOp(op, select);
    return success();
  }
};

// Lower stablehlo.or using De Morgan's law: a || b = !(!a && !b)
struct LowerOrOp : public OpRewritePattern<OrOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(OrOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // !a
    auto notLhs = NotOp::create(rewriter, loc, lhs);
    // !b
    auto notRhs = NotOp::create(rewriter, loc, rhs);
    // !a && !b
    auto andOp = AndOp::create(rewriter, loc, notLhs, notRhs);
    // !(!a && !b)
    auto result = NotOp::create(rewriter, loc, andOp);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower stablehlo.xor: a ^ b = (a && !b) || (!a && b)
// Using De Morgan: = !(!(a && !b) && !(!a && b))
struct LowerXorOp : public OpRewritePattern<XorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(XorOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    // !b
    auto notRhs = NotOp::create(rewriter, loc, rhs);
    // !a
    auto notLhs = NotOp::create(rewriter, loc, lhs);
    // a && !b
    auto lhsAndNotRhs = AndOp::create(rewriter, loc, lhs, notRhs);
    // !a && b
    auto notLhsAndRhs = AndOp::create(rewriter, loc, notLhs, rhs);
    // !(a && !b)
    auto notLhsAndNotRhs = NotOp::create(rewriter, loc, lhsAndNotRhs);
    // !(!a && b)
    auto notNotLhsAndRhs = NotOp::create(rewriter, loc, notLhsAndRhs);
    // !(a && !b) && !(!a && b)
    auto andOp = AndOp::create(rewriter, loc, notLhsAndNotRhs, notNotLhsAndRhs);
    // !(!(a && !b) && !(!a && b))
    auto result = NotOp::create(rewriter, loc, andOp);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower stablehlo.negate to 0 - x
struct LowerNegateOp : public OpRewritePattern<NegOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NegOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // Create zero constant with same shape and element type
    auto elemType = resultType.getElementType();
    Attribute zeroAttr;
    if (elemType.isIntOrIndex()) {
      zeroAttr = rewriter.getIntegerAttr(elemType, 0);
    } else if (isa<FloatType>(elemType)) {
      zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    } else {
      return failure();
    }
    
    auto denseAttr = DenseElementsAttr::get(resultType, zeroAttr);
    auto zero = ConstantOp::create(rewriter, loc, denseAttr);
    
    // 0 - operand
    auto result = rewriter.replaceOpWithNewOp<SubtractOp>(op, zero, operand);
    return success();
  }
};

// Lower stablehlo.abs to select(x >= 0, x, -x)
struct LowerAbsOp : public OpRewritePattern<AbsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AbsOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // Create zero constant
    auto elemType = resultType.getElementType();
    Attribute zeroAttr;
    if (elemType.isIntOrIndex()) {
      zeroAttr = rewriter.getIntegerAttr(elemType, 0);
    } else if (isa<FloatType>(elemType)) {
      zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
    } else {
      return failure();
    }
    
    auto denseAttr = DenseElementsAttr::get(resultType, zeroAttr);
    auto zero = ConstantOp::create(rewriter, loc, denseAttr);
    
    // operand >= 0
    ComparisonType compareType = isa<FloatType>(elemType) 
        ? ComparisonType::FLOAT : ComparisonType::SIGNED;
    auto compareTypeAttr = ComparisonTypeAttr::get(rewriter.getContext(), compareType);
    auto compare = CompareOp::create(
        rewriter, loc,
        RankedTensorType::get(resultType.getShape(), rewriter.getI1Type()),
        operand, zero,
        ComparisonDirection::GE,
        compareTypeAttr);
    
    // -operand (0 - operand)
    auto negOperand = rewriter.create<SubtractOp>(loc, zero, operand);
    
    // select(operand >= 0, operand, -operand)
    rewriter.replaceOpWithNewOp<SelectOp>(op, compare, operand, negOperand);
    return success();
  }
};

// Helper to create zero constant
Value createZeroConstant(PatternRewriter &rewriter, Location loc, RankedTensorType type) {
  auto elemType = type.getElementType();
  Attribute zeroAttr;
  if (elemType.isIntOrIndex()) {
    zeroAttr = rewriter.getIntegerAttr(elemType, 0);
  } else if (isa<FloatType>(elemType)) {
    zeroAttr = rewriter.getFloatAttr(elemType, 0.0);
  } else {
    return nullptr;
  }
  auto denseAttr = DenseElementsAttr::get(type, zeroAttr);
  return ConstantOp::create(rewriter, loc, denseAttr);
}

// Helper to create one constant
Value createOneConstant(PatternRewriter &rewriter, Location loc, RankedTensorType type) {
  auto elemType = type.getElementType();
  Attribute oneAttr;
  if (elemType.isIntOrIndex()) {
    oneAttr = rewriter.getIntegerAttr(elemType, 1);
  } else if (isa<FloatType>(elemType)) {
    oneAttr = rewriter.getFloatAttr(elemType, 1.0);
  } else {
    return nullptr;
  }
  auto denseAttr = DenseElementsAttr::get(type, oneAttr);
  return ConstantOp::create(rewriter, loc, denseAttr);
}

// Lower stablehlo.sign: sign(x) = select(x > 0, 1, select(x < 0, -1, 0))
struct LowerSignOp : public OpRewritePattern<SignOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SignOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    auto zero = createZeroConstant(rewriter, loc, resultType);
    if (!zero) return failure();
    auto one = createOneConstant(rewriter, loc, resultType);
    if (!one) return failure();
    auto negOne = rewriter.create<SubtractOp>(loc, zero, one);
    
    auto elemType = resultType.getElementType();
    ComparisonType compareType = isa<FloatType>(elemType) 
        ? ComparisonType::FLOAT : ComparisonType::SIGNED;
    auto compareTypeAttr = ComparisonTypeAttr::get(rewriter.getContext(), compareType);
    
    // x > 0
    auto gtZeroType = RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    auto gtZero = rewriter.create<CompareOp>(
        loc, gtZeroType, operand, zero,
        ComparisonDirection::GT, compareTypeAttr);
    
    // x < 0
    auto ltZeroType = RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    auto ltZero = rewriter.create<CompareOp>(
        loc, ltZeroType, operand, zero,
        ComparisonDirection::LT, compareTypeAttr);
    
    // select(x < 0, -1, 0)
    auto negOrZero = SelectOp::create(rewriter, loc, resultType, ltZero, negOne, zero);
    // select(x > 0, 1, select(x < 0, -1, 0))
    auto result = SelectOp::create(rewriter, loc, resultType, gtZero, one, negOrZero);
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower stablehlo.rsqrt to 1 / sqrt(x) using CustomCall for sqrt
struct LowerRsqrtOp : public OpRewritePattern<RsqrtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RsqrtOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // Create CustomCall for sqrt
    auto sqrtOp = CustomCallOp::create(
        rewriter, loc,
        resultType,
        ValueRange{operand},
        llvm::SmallVector<NamedAttribute>{
            rewriter.getNamedAttr("call_target_name", rewriter.getStringAttr("sqrt"))
        });
    Value sqrtResult = sqrtOp.getResult(0);
    
    // 1 / sqrt(x)
    auto one = createOneConstant(rewriter, loc, resultType);
    if (!one) return failure();
    rewriter.replaceOpWithNewOp<DivOp>(op, one, sqrtResult);
    return success();
  }
};

// Lower stablehlo.clamp to max(min(x, high), low)
// Since MinOp and MaxOp will be lowered, we lower clamp directly to select+compare
struct LowerClampOp : public OpRewritePattern<ClampOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ClampOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    Value min = op.getMin();
    Value max = op.getMax();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // Determine comparison type
    ComparisonType compareType = resultType.getElementType().isUnsignedInteger() 
        ? ComparisonType::UNSIGNED : ComparisonType::SIGNED;
    auto compareTypeAttr = ComparisonTypeAttr::get(rewriter.getContext(), compareType);
    auto boolType = RankedTensorType::get(resultType.getShape(), rewriter.getI1Type());
    
    // min(x, max) = select(x < max, x, max)
    auto ltMax = CompareOp::create(rewriter, loc, boolType, operand, max,
                                    ComparisonDirection::LT, compareTypeAttr);
    auto minOp = SelectOp::create(rewriter, loc, resultType, ltMax, operand, max);
    
    // max(min(x, max), min) = select(min(x, max) > min, min(x, max), min)
    auto gtMin = CompareOp::create(rewriter, loc, boolType, minOp, min,
                                    ComparisonDirection::GT, compareTypeAttr);
    auto result = SelectOp::create(rewriter, loc, resultType, gtMin, minOp, min);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Lower unary math operations to CustomCall for C library functions
template<typename OpType>
struct LowerUnaryMathOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  const char* funcName;
  
  LowerUnaryMathOp(MLIRContext* context, const char* name) 
      : OpRewritePattern<OpType>(context), funcName(name) {}

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    auto customCallOp = CustomCallOp::create(
        rewriter, loc,
        resultType,
        ValueRange{operand},
        llvm::SmallVector<NamedAttribute>{
            rewriter.getNamedAttr("call_target_name", rewriter.getStringAttr(funcName))
        });
    
    rewriter.replaceOp(op, customCallOp.getResults());
    return success();
  }
};

// Lower binary math operations to CustomCall
template<typename OpType>
struct LowerBinaryMathOp : public OpRewritePattern<OpType> {
  using OpRewritePattern<OpType>::OpRewritePattern;
  const char* funcName;
  
  LowerBinaryMathOp(MLIRContext* context, const char* name) 
      : OpRewritePattern<OpType>(context), funcName(name) {}

  LogicalResult matchAndRewrite(OpType op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    auto customCallOp = CustomCallOp::create(
        rewriter, loc,
        resultType,
        ValueRange{lhs, rhs},
        llvm::SmallVector<NamedAttribute>{
            rewriter.getNamedAttr("call_target_name", rewriter.getStringAttr(funcName))
        });
    
    rewriter.replaceOp(op, customCallOp.getResults());
    return success();
  }
};

// Lower stablehlo.logistic: 1 / (1 + exp(-x))
struct LowerLogisticOp : public OpRewritePattern<LogisticOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LogisticOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // -x
    auto zero = createZeroConstant(rewriter, loc, resultType);
    if (!zero) return failure();
    auto negX = rewriter.create<SubtractOp>(loc, zero, operand);
    
    // exp(-x)
    auto expNegXOp = CustomCallOp::create(
        rewriter, loc,
        resultType,
        ValueRange{negX},
        llvm::SmallVector<NamedAttribute>{
            rewriter.getNamedAttr("call_target_name", rewriter.getStringAttr("exp"))
        });
    Value expNegX = expNegXOp.getResult(0);
    
    // 1 + exp(-x)
    auto one = createOneConstant(rewriter, loc, resultType);
    if (!one) return failure();
    // AddOp::create(builder, loc, Type result, lhs, rhs) - need explicit result type
    auto onePlusExp = AddOp::create(rewriter, loc, resultType, one, expNegX);
    
    // 1 / (1 + exp(-x))
    rewriter.replaceOpWithNewOp<DivOp>(op, one, onePlusExp);
    return success();
  }
};

// Lower stablehlo.expm1: exp(x) - 1
struct LowerExpm1Op : public OpRewritePattern<Expm1Op> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Expm1Op op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value operand = op.getOperand();
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultType) return failure();
    
    // exp(x)
    auto expXOp = CustomCallOp::create(
        rewriter, loc,
        resultType,
        ValueRange{operand},
        llvm::SmallVector<NamedAttribute>{
            rewriter.getNamedAttr("call_target_name", rewriter.getStringAttr("exp"))
        });
    Value expX = expXOp.getResult(0);
    
    // exp(x) - 1
    auto one = createOneConstant(rewriter, loc, resultType);
    if (!one) return failure();
    rewriter.replaceOpWithNewOp<SubtractOp>(op, expX, one);
    return success();
  }
};

// Pass implementation is now in createLowerToSupportedOpsPass() to avoid TypeID issues

} // namespace

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createLowerToSupportedOpsPass() {
  struct LowerToSupportedOpsPassImpl : public PassWrapper<LowerToSupportedOpsPassImpl, OperationPass<ModuleOp>> {
    void runOnOperation() override {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      
      // Add lowering patterns for operations that can be expressed with supported ops
      patterns.add<LowerMinimumOp>(context);
      patterns.add<LowerOrOp>(context);
      patterns.add<LowerXorOp>(context);
      patterns.add<LowerNegateOp>(context);
      patterns.add<LowerAbsOp>(context);
      patterns.add<LowerSignOp>(context);
      patterns.add<LowerRsqrtOp>(context);
      patterns.add<LowerClampOp>(context);
      patterns.add<LowerLogisticOp>(context);
      patterns.add<LowerExpm1Op>(context);
      
      // Add lowering patterns for unary math operations (to CustomCall)
      patterns.add<LowerUnaryMathOp<SqrtOp>>(context, "sqrt");
      patterns.add<LowerUnaryMathOp<ExpOp>>(context, "exp");
      patterns.add<LowerUnaryMathOp<LogOp>>(context, "log");
      patterns.add<LowerUnaryMathOp<FloorOp>>(context, "floor");
      patterns.add<LowerUnaryMathOp<CeilOp>>(context, "ceil");
      patterns.add<LowerUnaryMathOp<RoundOp>>(context, "round");
      patterns.add<LowerUnaryMathOp<SineOp>>(context, "sin");
      patterns.add<LowerUnaryMathOp<CosineOp>>(context, "cos");
      patterns.add<LowerUnaryMathOp<TanOp>>(context, "tan");
      patterns.add<LowerUnaryMathOp<TanhOp>>(context, "tanh");
      patterns.add<LowerUnaryMathOp<CbrtOp>>(context, "cbrt");
      patterns.add<LowerUnaryMathOp<Log1pOp>>(context, "log1p");
      patterns.add<LowerUnaryMathOp<IsFiniteOp>>(context, "isfinite");
      
      // Add lowering patterns for binary math operations (to CustomCall)
      patterns.add<LowerBinaryMathOp<PowOp>>(context, "pow");
      patterns.add<LowerBinaryMathOp<Atan2Op>>(context, "atan2");
      
      // Apply patterns greedily
      GreedyRewriteConfig config;
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
        signalPassFailure();
      }
    }
    
    StringRef getArgument() const override { return "lower-to-supported-ops"; }
    StringRef getDescription() const override {
      return "Lower unsupported StableHLO operations to supported ones";
    }
  };
  
  return std::make_unique<LowerToSupportedOpsPassImpl>();
}

} // namespace stablehlo
} // namespace mlir

