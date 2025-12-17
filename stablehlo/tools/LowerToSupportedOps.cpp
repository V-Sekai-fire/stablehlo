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
    auto select = SelectOp::create(rewriter, loc, compare, lhs, rhs);
    
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
    auto result = SubtractOp::create(rewriter, loc, zero, operand);
    
    rewriter.replaceOp(op, result);
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
    auto negOperand = SubtractOp::create(rewriter, loc, zero, operand);
    
    // select(operand >= 0, operand, -operand)
    auto result = SelectOp::create(rewriter, loc, compare, operand, negOperand);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

// Pass to lower unsupported operations to supported ones
struct LowerToSupportedOpsPass : public PassWrapper<LowerToSupportedOpsPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    
    // Add lowering patterns
    patterns.add<LowerMinimumOp>(context);
    patterns.add<LowerOrOp>(context);
    patterns.add<LowerXorOp>(context);
    patterns.add<LowerNegateOp>(context);
    patterns.add<LowerAbsOp>(context);
    
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

} // namespace

namespace mlir {
namespace stablehlo {

std::unique_ptr<Pass> createLowerToSupportedOpsPass() {
  return std::make_unique<LowerToSupportedOpsPass>();
}

} // namespace stablehlo
} // namespace mlir

