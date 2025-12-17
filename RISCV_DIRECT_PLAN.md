# 100% StableHLO to RISC-V ELF64 Direct Codegen Plan

## Executive Summary

This document provides a plan to achieve **100% support** for all 115 StableHLO operations in **direct RISC-V ELF64 codegen** via the MLIR → LLVM pipeline, **dropping C/C++ codegen entirely**.

**Current Status**: RISC-V codegen exists but may have unsupported operations  
**Target**: 115/115 operations (100%) via direct RISC-V compilation  
**Approach**: Fix MLIR lowering pipeline, not generate C/C++ code

---

## Current Architecture

### RISC-V Codegen Pipeline (Already Implemented)

```
StableHLO MLIR
    ↓
StableHLO → Linalg (conversion pass)
    ↓
Bufferization (tensors → memrefs)
    ↓
Linalg → Loops (lowering)
    ↓
SCF → CF (control flow)
    ↓
Affine → Standard (affine ops)
    ↓
Standard → LLVM Dialect (conversion)
    ↓
LLVM Dialect → LLVM IR (translation)
    ↓
LLVM IR → RISC-V ELF64 (codegen)
```

**Status**: ✅ Pipeline exists and works for supported operations

---

## Implementation Strategy

### Core Principles

1. **Direct RISC-V**: No C/C++ intermediate code
2. **MLIR Pipeline**: Fix lowering passes to support all operations
3. **LLVM Backend**: Leverage LLVM's RISC-V codegen
4. **Incremental**: Fix operations one by one through the pipeline
5. **Testing**: Validate with RISC-V execution or simulation

### Key Insight

The RISC-V codegen doesn't need C/C++ codegen at all - it goes directly through MLIR lowering passes to LLVM IR, then LLVM generates RISC-V code. The work is:

1. **Ensure all StableHLO ops can be lowered** through the MLIR pipeline
2. **Fix any lowering issues** in the conversion passes
3. **Add missing lowering patterns** for unsupported operations
4. **Test RISC-V codegen** for each operation

---

## Operation Support Analysis

### Operations That Should Already Work

These operations should work through the existing MLIR lowering pipeline:

1. **Element-wise operations**: add, multiply, subtract, divide, etc.

   - Lowered to: Linalg generic ops → loops → LLVM
   - **Status**: ✅ Should work

2. **Math functions**: sqrt, exp, log, sin, cos, etc.

   - Lowered to: LLVM intrinsics or library calls
   - **Status**: ✅ Should work (may need runtime library)

3. **Shape operations**: reshape, transpose, slice, etc.

   - Lowered to: Memref operations → LLVM
   - **Status**: ⚠️ May need fixes

4. **Linear algebra**: dot, einsum, etc.
   - Lowered to: Linalg ops → loops → LLVM
   - **Status**: ⚠️ May need optimization

### Operations That Need Work

1. **Control flow**: if, while, case, map

   - **Fix**: Ensure SCF → CF conversion handles all cases
   - **Status**: ⚠️ Needs verification

2. **Collective operations**: all_gather, all_reduce, etc.

   - **Fix**: Lower to runtime calls or stubs
   - **Status**: ❌ Needs implementation

3. **Communication**: send, recv, infeed, outfeed

   - **Fix**: Lower to runtime calls or stubs
   - **Status**: ❌ Needs implementation

4. **Complex numbers**: complex, real, imag

   - **Fix**: Lower to struct types in LLVM
   - **Status**: ⚠️ Needs verification

5. **Special operations**: FFT, RNG, sort
   - **Fix**: Lower to library calls or implement
   - **Status**: ⚠️ Needs work

---

## Implementation Plan

### Phase 1: Audit Current Pipeline (Week 1)

**Goal**: Understand what works and what doesn't

**Tasks**:

1. Test all 115 StableHLO operations with RISC-V codegen
2. Identify which operations fail
3. Categorize failures:
   - Missing lowering patterns
   - Incorrect lowering
   - Runtime library needs
   - Unsupported semantics

**Deliverable**: List of unsupported operations with failure reasons

---

### Phase 2: Fix Basic Lowering Issues (Weeks 2-4)

**Goal**: Fix operations that should work but don't

**Tasks**:

1. Fix shape operation lowering (reshape, transpose, slice)
2. Fix type conversion operations
3. Fix bit operations (shift, clz, popcount)
4. Fix comparison operations
5. Test each fix with RISC-V codegen

**Coverage Target**: +20-30 operations

---

### Phase 3: Implement Missing Lowering Patterns (Weeks 5-10)

**Goal**: Add lowering patterns for operations that need them

**Tasks**:

1. **Control Flow** (Week 5-6):

   - Fix `stablehlo.if` → SCF if
   - Fix `stablehlo.while` → SCF while
   - Fix `stablehlo.case` → SCF switch
   - Fix `stablehlo.map` → loops

2. **Complex Numbers** (Week 7):

   - Lower complex types to LLVM structs
   - Implement complex operations

3. **Tensor Operations** (Week 8-9):

   - Fix gather/scatter operations
   - Fix pad/concatenate operations
   - Fix broadcast operations

4. **Linear Algebra** (Week 10):
   - Optimize dot product lowering
   - Fix einsum lowering
   - Add Cholesky/triangular solve (via library calls)

**Coverage Target**: +40-50 operations

---

### Phase 4: Runtime Library Integration (Weeks 11-14)

**Goal**: Handle operations that need runtime support

**Tasks**:

1. **Math Functions** (Week 11):

   - Ensure math library linking works
   - Test all math functions on RISC-V

2. **Special Operations** (Week 12-13):

   - FFT: Link FFT library or implement
   - RNG: Implement RNG or link library
   - Sort: Implement or use library

3. **Batch Normalization** (Week 14):
   - Lower to element-wise operations
   - Test correctness

**Coverage Target**: +15-20 operations

---

### Phase 5: Distributed/Communication Operations (Weeks 15-18)

**Goal**: Handle operations requiring runtime infrastructure

**Tasks**:

1. **Collective Operations** (Week 15-17):

   - Lower to runtime stubs (single-device mode)
   - Document limitations
   - Add runtime hooks for future multi-device support

2. **Communication Operations** (Week 18):
   - Lower to I/O stubs
   - Document I/O requirements
   - Add file/stdin/stdout support

**Coverage Target**: +11 operations (stub implementations)

---

### Phase 6: Testing & Validation (Weeks 19-20)

**Goal**: Ensure all operations work correctly

**Tasks**:

1. Create test suite for all 115 operations
2. Test on RISC-V simulator (QEMU, Spike, etc.)
3. Validate correctness against StableHLO interpreter
4. Performance benchmarking
5. Documentation

**Coverage Target**: 115/115 operations (100%)

---

## Technical Approach

### 1. Lowering Pattern Development

For each unsupported operation, create a lowering pattern:

```cpp
// Example: Lower stablehlo.reshape to memref operations
class LowerReshapeOp : public OpRewritePattern<stablehlo::ReshapeOp> {
  LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp op,
      PatternRewriter &rewriter) const override {
    // Lower reshape to memref operations
    // Reshape is just a view change, so use memref operations
    auto input = op.getOperand();
    auto outputType = op.getType();

    // Create memref operations for reshape
    // ...

    rewriter.replaceOp(op, result);
    return success();
  }
};
```

### 2. Runtime Library Strategy

For operations needing runtime support:

1. **Math Functions**: Link against libm (standard C math library)
2. **FFT**: Link against FFTW (GPL) or KissFFT (BSD) or implement
3. **RNG**: Implement simple RNG or link against library
4. **BLAS**: Link against OpenBLAS (optional, for performance)

### 3. Stub Implementations

For operations that can't be fully implemented:

1. **Collective Ops**: Generate stubs that work for single-device
2. **Communication Ops**: Generate I/O stubs
3. **Document Limitations**: Clearly mark what's supported

---

## Comparison: C/C++ Codegen vs Direct RISC-V

| Aspect                | C/C++ Codegen         | Direct RISC-V               |
| --------------------- | --------------------- | --------------------------- |
| **Intermediate Code** | C/C++ source          | LLVM IR                     |
| **Compiler Needed**   | C/C++ compiler        | LLVM (already in pipeline)  |
| **Optimization**      | Depends on C compiler | LLVM optimizations          |
| **Binary Size**       | Larger (C runtime)    | Smaller (direct)            |
| **Performance**       | Good                  | Better (LLVM optimizations) |
| **Complexity**        | High (manual codegen) | Lower (MLIR passes)         |
| **Maintenance**       | High (manual updates) | Lower (MLIR ecosystem)      |

**Verdict**: Direct RISC-V is **better** - simpler, faster, better optimized.

---

## Estimated Timeline

- **Phase 1** (Audit): 1 week
- **Phase 2** (Basic fixes): 3 weeks
- **Phase 3** (Missing patterns): 6 weeks
- **Phase 4** (Runtime): 4 weeks
- **Phase 5** (Distributed): 4 weeks
- **Phase 6** (Testing): 2 weeks

**Total**: ~20 weeks (~5 months) vs 44 weeks for C/C++ codegen

**Much faster** because:

- Leverages existing MLIR infrastructure
- No manual code generation
- LLVM handles optimization
- Reuses existing lowering passes

---

## Success Criteria

1. ✅ All 115 StableHLO operations compile to RISC-V ELF64
2. ✅ Generated code executes correctly on RISC-V
3. ✅ Performance is acceptable (LLVM optimizations)
4. ✅ Test suite passes for all operations
5. ✅ Documentation complete

---

## Next Steps

1. **Immediate**: Audit current RISC-V codegen support
2. **Week 1**: Identify all unsupported operations
3. **Week 2+**: Fix operations systematically
4. **Ongoing**: Test with RISC-V simulator

---

## Conclusion

Dropping C/C++ codegen and focusing on **direct RISC-V codegen** is the right approach:

- ✅ **Simpler**: Use existing MLIR pipeline
- ✅ **Faster**: ~20 weeks vs 44 weeks
- ✅ **Better**: LLVM optimizations
- ✅ **Maintainable**: MLIR ecosystem support

The work is primarily about **fixing MLIR lowering passes**, not writing code generators.
