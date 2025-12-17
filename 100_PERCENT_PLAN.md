# 100% StableHLO Operation Support Implementation Plan

## Executive Summary

This document provides a comprehensive plan to achieve **100% support** for all 115 StableHLO operations in the C99 codegen backend, compatible with TinyCC and using non-GPL FOSS libraries.

**Current Status**: 38/115 operations (33%)  
**Target**: 115/115 operations (100%)  
**Remaining**: 77 operations (67%)

---

## Implementation Strategy

### Core Principles

1. **TinyCC Compatibility**: All generated code must compile with TinyCC
2. **Non-GPL FOSS**: Only use permissively licensed libraries (MIT, BSD, Apache, Boost)
3. **Incremental Progress**: Implement in phases, testing after each phase
4. **Performance**: Optimize with SIMDe/SLEEF where possible
5. **Correctness**: Maintain semantic correctness for all operations

### Libraries & Tools

| Library      | License      | Purpose         | Status                                       |
| ------------ | ------------ | --------------- | -------------------------------------------- |
| **SIMDe**    | MIT          | SIMD operations | ✅ To be integrated                          |
| **SLEEF**    | BSL-1.0      | Math functions  | ✅ To be integrated                          |
| **OpenBLAS** | BSD-3-Clause | Linear algebra  | ⚠️ Optional (recommended)                    |
| **GotoBLAS** | BSD          | Linear algebra  | ❌ Not recommended (unmaintained since 2008) |
| **XNNPACK**  | BSD-3-Clause | Convolution     | ⚠️ Optional (pre-compiled)                   |

**Note on BLAS Libraries**:

- **GotoBLAS**: Original BSD-licensed BLAS implementation by Kazushige Goto. Development ceased in 2008, optimized only for Intel Nehalem architecture. Not recommended due to lack of maintenance and outdated optimizations.
- **OpenBLAS**: Actively maintained fork of GotoBLAS2, also BSD-3-Clause licensed. Includes modern optimizations for current architectures (x86, ARM, RISC-V). **Recommended choice** for linear algebra operations.

---

## Operation Categories & Implementation Plan

### Phase 1: Foundation & Optimization (Weeks 1-4)

**Goal**: Optimize existing operations and add foundational infrastructure

#### 1.1 SIMDe Integration (Week 1-2)

- **Operations**: Optimize 28 element-wise operations
- **Tasks**:
  - Add SIMDe as dependency (CMake FetchContent)
  - Update `COpEmitters.cpp` to use SIMDe for vectorized operations
  - Generate SIMD loops with scalar fallback
  - Test with TinyCC compilation
- **Expected Improvement**: 2-4x performance for element-wise ops
- **Coverage**: 28/28 operations (already supported, performance only)

#### 1.2 SLEEF Integration (Week 2-3)

- **Operations**: Optimize 15 math function operations
- **Tasks**:
  - Add SLEEF as dependency
  - Update `LowerToSupportedOps.cpp` to use SLEEF functions
  - Generate vectorized math function calls
  - Test accuracy and performance
- **Expected Improvement**: 4-8x performance for math functions
- **Coverage**: 15/15 operations (already supported, performance only)

#### 1.3 Bit Operations (Week 3-4)

- **Operations**: 5 operations
  1. `stablehlo.shift_left`
  2. `stablehlo.shift_right_arithmetic`
  3. `stablehlo.shift_right_logical`
  4. `stablehlo.clz` (count leading zeros)
  5. `stablehlo.population_count`
- **Implementation**:
  - Use C bitwise operators (`<<`, `>>`, `&`, `|`)
  - Use SIMDe for vectorized bit operations where possible
  - Implement `clz` using compiler intrinsics or lookup tables
  - Implement `population_count` using bit manipulation tricks
- **Coverage**: +5 operations (5/5 = 100% of bit ops)
- **Total After Phase 1**: 43/115 (37%)

---

### Phase 2: Shape & Tensor Manipulation (Weeks 5-10)

**Goal**: Implement all shape and tensor manipulation operations

#### 2.1 Basic Shape Operations (Week 5-6)

- **Operations**: 4 operations
  1. `stablehlo.reshape` - Reshape tensor (view change)
  2. `stablehlo.transpose` - Transpose dimensions
  3. `stablehlo.reverse` - Reverse dimensions
  4. `stablehlo.get_dimension_size` - Get dimension size
- **Implementation**:
  - `reshape`: Simple memcpy (same data, different view)
  - `transpose`: Index calculation with nested loops
  - `reverse`: Index manipulation per dimension
  - `get_dimension_size`: Return shape array element
- **Coverage**: +4 operations

#### 2.2 Broadcasting Operations (Week 6-7)

- **Operations**: 3 operations
  1. `stablehlo.broadcast` - Broadcast tensor dimensions
  2. `stablehlo.broadcast_in_dim` - Broadcast with dimension mapping
  3. `stablehlo.dynamic_broadcast_in_dim` - Dynamic broadcast
- **Implementation**:
  - Calculate output shape from input shape and broadcast dimensions
  - Generate nested loops with index calculation
  - Handle dimension mapping for `broadcast_in_dim`
  - Support dynamic shapes for `dynamic_broadcast_in_dim`
- **Coverage**: +3 operations

#### 2.3 Slice Operations (Week 7-8)

- **Operations**: 4 operations
  1. `stablehlo.slice` - Slice tensor
  2. `stablehlo.dynamic_slice` - Dynamic slice
  3. `stablehlo.dynamic_update_slice` - Dynamic update slice
  4. `stablehlo.real_dynamic_slice` - Real dynamic slice
- **Implementation**:
  - Calculate slice indices from start/limit/strides
  - Generate loops with computed indices
  - Handle dynamic indices (runtime calculation)
  - Support update operations (write to slice)
- **Coverage**: +4 operations

#### 2.4 Padding & Concatenation (Week 8-9)

- **Operations**: 3 operations
  1. `stablehlo.pad` - Pad tensor
  2. `stablehlo.dynamic_pad` - Dynamic pad
  3. `stablehlo.concatenate` - Concatenate tensors
- **Implementation**:
  - `pad`: Initialize output, copy input with offset, fill padding
  - `dynamic_pad`: Calculate padding sizes at runtime
  - `concatenate`: Copy multiple tensors to output along specified dimension
- **Coverage**: +3 operations

#### 2.5 Dynamic Reshape (Week 9-10)

- **Operations**: 1 operation
  1. `stablehlo.dynamic_reshape` - Dynamic reshape
- **Implementation**:
  - Calculate output shape at runtime
  - Validate total element count matches
  - Copy data (same as static reshape)
- **Coverage**: +1 operation

**Total After Phase 2**: 58/115 (50%)

---

### Phase 3: Linear Algebra (Weeks 11-14)

**Goal**: Implement matrix operations and linear algebra

#### 3.1 Basic Matrix Operations (Week 11-12)

- **Operations**: 2 operations
  1. `stablehlo.dot` - Matrix multiplication
  2. `stablehlo.dot_general` - General dot product
- **Implementation Options**:
  - **Option A**: Manual implementation (nested loops)
  - **Option B**: OpenBLAS CBLAS interface (recommended for performance)
  - **Option C**: GotoBLAS CBLAS interface (not recommended - unmaintained)
  - **Option D**: XNNPACK (if pre-compiled)
- **Recommendation**: Start with manual, add OpenBLAS as optional optimization
- **Note**: GotoBLAS is BSD-licensed but unmaintained since 2008. OpenBLAS is the modern, actively maintained alternative with better performance on current hardware.
- **Coverage**: +2 operations

#### 3.2 Advanced Linear Algebra (Week 12-13)

- **Operations**: 2 operations
  1. `stablehlo.einsum` - Einstein summation
  2. `stablehlo.unary_einsum` - Unary einsum
- **Implementation**:
  - Parse einsum equation
  - Generate nested loops based on equation
  - Handle dimension reduction and broadcasting
- **Coverage**: +2 operations

#### 3.3 Matrix Decomposition (Week 13-14)

- **Operations**: 2 operations
  1. `stablehlo.cholesky` - Cholesky decomposition
  2. `stablehlo.triangular_solve` - Triangular solve
- **Implementation**:
  - Use LAPACKE (C interface to LAPACK) via OpenBLAS (recommended)
  - GotoBLAS also includes LAPACK but is unmaintained
  - Or implement manually (more complex)
- **Coverage**: +2 operations

#### 3.4 Reduce Window (Week 14)

- **Operations**: 1 operation
  1. `stablehlo.reduce_window` - Reduce window operation
- **Implementation**:
  - Sliding window over input tensor
  - Apply reduction function (add, max, etc.) to each window
  - Handle padding and strides
- **Coverage**: +1 operation

**Total After Phase 3**: 67/115 (58%)

---

### Phase 4: Data Gathering & Scattering (Weeks 15-18)

**Goal**: Implement gather, scatter, and indexing operations

#### 4.1 Gather Operations (Week 15-16)

- **Operations**: 3 operations
  1. `stablehlo.gather` - Gather operation
  2. `stablehlo.dynamic_gather` - Dynamic gather
  3. `stablehlo.torch_index_select` - Torch-style index select
- **Implementation**:
  - Use index tensor to select elements from input
  - Generate loops over output shape
  - Calculate source indices from index tensor
  - Handle dynamic indices
- **Coverage**: +3 operations

#### 4.2 Scatter Operations (Week 16-17)

- **Operations**: 2 operations
  1. `stablehlo.scatter` - Scatter operation
  2. `stablehlo.select_and_scatter` - Select and scatter
- **Implementation**:
  - Initialize output tensor
  - Use index tensor to determine write locations
  - Apply update function (add, max, etc.)
  - Handle duplicate indices (combine updates)
- **Coverage**: +2 operations

#### 4.3 Reduce Scatter (Week 17-18)

- **Operations**: 1 operation
  1. `stablehlo.reduce_scatter` - Reduce scatter
- **Implementation**:
  - Similar to scatter but with reduction across replicas
  - For single-device: treat as regular scatter with reduction
  - Note: Full implementation requires distributed runtime
- **Coverage**: +1 operation (partial, single-device only)

**Total After Phase 4**: 73/115 (63%)

---

### Phase 5: Control Flow (Weeks 19-21)

**Goal**: Implement control flow operations

#### 5.1 Conditional Execution (Week 19)

- **Operations**: 2 operations
  1. `stablehlo.if` - Conditional execution
  2. `stablehlo.case` - Case/switch statement
- **Implementation**:
  - Generate C `if`/`else` statements
  - Generate C `switch` statements
  - Handle function calls for true/false branches
  - Support multiple case branches
- **Coverage**: +2 operations

#### 5.2 Loops (Week 20)

- **Operations**: 1 operation
  1. `stablehlo.while` - While loop
- **Implementation**:
  - Generate C `while` loop
  - Evaluate condition function
  - Execute body function
  - Handle loop-carried values
- **Coverage**: +1 operation

#### 5.3 Map Operation (Week 21)

- **Operations**: 1 operation
  1. `stablehlo.map` - Map operation
- **Implementation**:
  - Generate loop over input dimensions
  - Apply function to each element
  - Handle broadcasting of function arguments
- **Coverage**: +1 operation

**Total After Phase 5**: 77/115 (67%)

---

### Phase 6: Type & Format Conversion (Weeks 22-24)

**Goal**: Implement type conversion and quantization

#### 6.1 Type Conversion (Week 22)

- **Operations**: 2 operations
  1. `stablehlo.convert` - Type conversion
  2. `stablehlo.bitcast_convert` - Bitcast conversion
- **Implementation**:
  - `convert`: Use C casts (e.g., `(float)int_value`)
  - `bitcast_convert`: Use `memcpy` with reinterpretation
  - Handle all type combinations (int↔float, different bit widths)
- **Coverage**: +2 operations

#### 6.2 Quantization (Week 23-24)

- **Operations**: 2 operations
  1. `stablehlo.uniform_quantize` - Uniform quantization
  2. `stablehlo.uniform_dequantize` - Uniform dequantization
- **Implementation**:
  - Apply quantization formula: `quantized = round((value - zero_point) / scale)`
  - Apply dequantization: `value = (quantized - zero_point) * scale`
  - Handle different quantization schemes
- **Coverage**: +2 operations

**Total After Phase 6**: 81/115 (70%)

---

### Phase 7: Complex Numbers (Weeks 25-26)

**Goal**: Implement complex number operations

#### 7.1 Complex Number Support (Week 25-26)

- **Operations**: 3 operations
  1. `stablehlo.complex` - Create complex number
  2. `stablehlo.real` - Real part
  3. `stablehlo.imag` - Imaginary part
- **Implementation**:
  - Define complex number struct:
    ```c
    typedef struct {
      float real;
      float imag;
    } complex_float;
    ```
  - `complex`: Combine real and imaginary parts
  - `real`: Extract real component
  - `imag`: Extract imaginary component
- **Coverage**: +3 operations

**Total After Phase 7**: 84/115 (73%)

---

### Phase 8: Special Operations (Weeks 27-32)

**Goal**: Implement special operations and utilities

#### 8.1 Sequence Generation (Week 27)

- **Operations**: 2 operations
  1. `stablehlo.iota` - Generate iota sequence
  2. `stablehlo.dynamic_iota` - Dynamic iota
- **Implementation**:
  - Generate sequence: `[0, 1, 2, ..., n-1]` along specified dimension
  - Calculate indices and assign values
  - Handle dynamic shapes
- **Coverage**: +2 operations

#### 8.2 Random Number Generation (Week 28)

- **Operations**: 2 operations
  1. `stablehlo.rng` - Random number generation
  2. `stablehlo.rng_bit_generator` - RNG bit generator
- **Implementation**:
  - Use C standard library `rand()` or better RNG (e.g., PCG)
  - Implement different distributions (uniform, normal)
  - Handle RNG state management
  - Note: May need external RNG library (non-GPL)
- **Coverage**: +2 operations

#### 8.3 Sorting (Week 29)

- **Operations**: 1 operation
  1. `stablehlo.sort` - Sort operation
- **Implementation**:
  - Use C standard library `qsort()` or implement quicksort
  - Handle multi-dimensional sorting
  - Support different comparison directions
- **Coverage**: +1 operation

#### 8.4 FFT (Week 30)

- **Operations**: 1 operation
  1. `stablehlo.fft` - Fast Fourier Transform
- **Implementation Options**:
  - **Option A**: Use FFTW (GPL, not acceptable)
  - **Option B**: Use KissFFT (BSD-3-Clause, acceptable)
  - **Option C**: Implement Cooley-Tukey FFT manually
- **Recommendation**: Use KissFFT or manual implementation
- **Coverage**: +1 operation

#### 8.5 Precision & Optimization (Week 31)

- **Operations**: 2 operations
  1. `stablehlo.reduce_precision` - Reduce precision
  2. `stablehlo.optimization_barrier` - Optimization barrier
- **Implementation**:
  - `reduce_precision`: Cast to lower precision and back
  - `optimization_barrier`: Use `volatile` or compiler barriers
- **Coverage**: +2 operations

#### 8.6 Rounding & Composite (Week 32)

- **Operations**: 2 operations
  1. `stablehlo.round_nearest_even` - Round to nearest even
  2. `stablehlo.composite` - Composite operation
- **Implementation**:
  - `round_nearest_even`: Implement banker's rounding
  - `composite`: Expand to constituent operations
- **Coverage**: +2 operations

**Total After Phase 8**: 94/115 (82%)

---

### Phase 9: Batch Normalization (Weeks 33-34)

**Goal**: Implement batch normalization operations

#### 9.1 Batch Normalization (Week 33-34)

- **Operations**: 3 operations
  1. `stablehlo.batch_norm_inference` - Batch norm inference
  2. `stablehlo.batch_norm_training` - Batch norm training
  3. `stablehlo.batch_norm_grad` - Batch norm gradient
- **Implementation**:
  - Apply batch normalization formula:
    ```c
    normalized = (input - mean) / sqrt(variance + epsilon)
    output = normalized * scale + offset
    ```
  - Handle training vs inference modes
  - Compute gradients for training
- **Coverage**: +3 operations

**Total After Phase 9**: 97/115 (84%)

---

### Phase 10: Advanced Operations (Weeks 35-38)

**Goal**: Implement remaining advanced operations

#### 10.1 Dynamic Convolution (Week 35)

- **Operations**: 1 operation
  1. `stablehlo.dynamic_conv` - Dynamic convolution
- **Implementation**:
  - Similar to static convolution but with dynamic padding/strides
  - Calculate output shape at runtime
  - Use same convolution algorithm as static version
- **Coverage**: +1 operation

#### 10.2 Tuple Operations (Week 36)

- **Operations**: 2 operations
  1. `stablehlo.tuple` - Create tuple
  2. `stablehlo.get_tuple_element` - Get tuple element
- **Implementation**:
  - Use C struct to represent tuples
  - `tuple`: Pack values into struct
  - `get_tuple_element`: Extract element from struct
- **Coverage**: +2 operations

#### 10.3 Token Operations (Week 37)

- **Operations**: 2 operations
  1. `stablehlo.create_token` - Create token
  2. `stablehlo.after_all` - After all tokens
- **Implementation**:
  - Tokens are ordering constraints (no-op in single-threaded)
  - `create_token`: Return dummy token value
  - `after_all`: Combine tokens (no-op)
  - Note: Full semantics require async execution
- **Coverage**: +2 operations

#### 10.4 Dimension Operations (Week 38)

- **Operations**: 1 operation
  1. `stablehlo.set_dimension_size` - Set dimension size
- **Implementation**:
  - Modify shape array at specified dimension
  - Validate new size is <= original size
  - Update shape metadata
- **Coverage**: +1 operation

**Total After Phase 10**: 103/115 (90%)

---

### Phase 11: Distributed & Communication (Weeks 39-42)

**Goal**: Implement distributed and communication operations

#### 11.1 Collective Operations (Week 39-41)

- **Operations**: 7 operations
  1. `stablehlo.all_gather` - All-gather collective
  2. `stablehlo.all_reduce` - All-reduce collective
  3. `stablehlo.all_to_all` - All-to-all collective
  4. `stablehlo.reduce_scatter` - Reduce-scatter collective
  5. `stablehlo.collective_broadcast` - Collective broadcast
  6. `stablehlo.collective_permute` - Collective permute
  7. `stablehlo.cross_replica_sum` - Cross-replica sum
- **Implementation Strategy**:
  - **Single-device mode**: Implement as no-ops or identity operations
  - **Multi-device mode**: Require distributed runtime (future work)
  - Generate stub implementations that work for single-device
  - Document limitations
- **Coverage**: +7 operations (partial, single-device only)

#### 11.2 Communication Operations (Week 41-42)

- **Operations**: 4 operations
  1. `stablehlo.send` - Send operation
  2. `stablehlo.recv` - Receive operation
  3. `stablehlo.infeed` - Infeed operation
  4. `stablehlo.outfeed` - Outfeed operation
- **Implementation Strategy**:
  - **Single-device mode**: Stub implementations
  - **I/O mode**: Use file I/O or stdin/stdout
  - Generate conditional code based on execution mode
  - Document I/O requirements
- **Coverage**: +4 operations (partial, requires I/O infrastructure)

**Total After Phase 11**: 114/115 (99%)

---

### Phase 12: Replica/Partition & Finalization (Weeks 43-44)

**Goal**: Complete remaining operations and finalize

#### 12.1 Replica/Partition Operations (Week 43)

- **Operations**: 2 operations
  1. `stablehlo.replica_id` - Get replica ID
  2. `stablehlo.partition_id` - Get partition ID
- **Implementation**:
  - Return constant 0 for single-device mode
  - Can be configured via function parameters
  - Document multi-device limitations
- **Coverage**: +2 operations

#### 12.2 Complete Reduce & Convolution (Week 44)

- **Operations**: 2 operations (already marked as TODO)
  1. `stablehlo.reduce` - Complete implementation
  2. `stablehlo.convolution` - Complete implementation
- **Implementation**:
  - `reduce`: Implement all reduction types (add, multiply, max, min, etc.)
  - `convolution`: Implement full convolution with all options
- **Coverage**: +0 operations (already counted, completing TODOs)

#### 12.3 Return Operation (Week 44)

- **Operations**: 1 operation
  1. `stablehlo.return` - Return operation
- **Implementation**:
  - Already handled by `func.return`
  - Add explicit support if needed
- **Coverage**: +1 operation

**Total After Phase 12**: 115/115 (100%) ✅

---

## Implementation Summary

### Coverage by Phase

| Phase        | Operations Added | Cumulative | Percentage  |
| ------------ | ---------------- | ---------- | ----------- |
| **Current**  | 38               | 38         | 33%         |
| **Phase 1**  | +5               | 43         | 37%         |
| **Phase 2**  | +15              | 58         | 50%         |
| **Phase 3**  | +9               | 67         | 58%         |
| **Phase 4**  | +6               | 73         | 63%         |
| **Phase 5**  | +4               | 77         | 67%         |
| **Phase 6**  | +4               | 81         | 70%         |
| **Phase 7**  | +3               | 84         | 73%         |
| **Phase 8**  | +10              | 94         | 82%         |
| **Phase 9**  | +3               | 97         | 84%         |
| **Phase 10** | +6               | 103        | 90%         |
| **Phase 11** | +11              | 114        | 99%         |
| **Phase 12** | +1               | 115        | **100%** ✅ |

### Timeline

- **Total Duration**: 44 weeks (~11 months)
- **Phases 1-6** (Foundation): 24 weeks (~6 months) → 81/115 (70%)
- **Phases 7-10** (Advanced): 14 weeks (~3.5 months) → 103/115 (90%)
- **Phases 11-12** (Distributed/Final): 6 weeks (~1.5 months) → 115/115 (100%)

### Resource Requirements

1. **Developer Time**: ~1 FTE for 11 months
2. **Testing**: Comprehensive test suite for each operation
3. **Dependencies**:
   - SIMDe (MIT) - Header-only
   - SLEEF (BSL-1.0) - Header-only
   - OpenBLAS (BSD-3-Clause) - Optional, for linear algebra
   - KissFFT (BSD-3-Clause) - Optional, for FFT
4. **Infrastructure**: CI/CD for testing with TinyCC

---

## Testing Strategy

### Unit Tests

- Test each operation with various input shapes
- Test edge cases (empty tensors, single elements, etc.)
- Test type combinations (int32, int64, float32, float64)

### Integration Tests

- Test operation combinations
- Test with real MLIR files from test suite
- Test compilation with TinyCC

### Performance Tests

- Benchmark SIMDe/SLEEF optimizations
- Compare against reference implementations
- Measure compilation time with TinyCC

### Correctness Tests

- Compare outputs against StableHLO reference implementation
- Test numerical accuracy for floating-point operations
- Validate shape calculations

---

## Risk Mitigation

### High-Risk Areas

1. **Distributed Operations** (Phase 11)

   - **Risk**: Require distributed runtime
   - **Mitigation**: Implement single-device stubs, document limitations

2. **Complex Operations** (Phase 8)

   - **Risk**: FFT, RNG may need external libraries
   - **Mitigation**: Use non-GPL alternatives (KissFFT, PCG)

3. **Performance** (All phases)

   - **Risk**: Generated code may be slow
   - **Mitigation**: Use SIMDe/SLEEF, optimize hot paths

4. **TinyCC Compatibility** (All phases)
   - **Risk**: Some C99 features may not work
   - **Mitigation**: Test early, use C89-compatible code where possible

---

## Success Criteria

### Phase Completion Criteria

- ✅ All operations in phase compile with TinyCC
- ✅ All operations pass unit tests
- ✅ Generated code is semantically correct
- ✅ Documentation updated

### Final Completion Criteria

- ✅ 115/115 operations supported (100%)
- ✅ All operations compile with TinyCC
- ✅ Comprehensive test suite passes
- ✅ Performance benchmarks meet targets
- ✅ Documentation complete

---

## Next Steps

1. **Immediate** (Week 1):

   - Set up SIMDe integration
   - Create test infrastructure
   - Begin Phase 1 implementation

2. **Short-term** (Weeks 1-4):

   - Complete Phase 1 (Foundation)
   - Establish development workflow
   - Set up CI/CD

3. **Medium-term** (Weeks 5-24):

   - Complete Phases 2-6 (Foundation)
   - Reach 70% coverage
   - Regular testing and validation

4. **Long-term** (Weeks 25-44):
   - Complete Phases 7-12 (Advanced)
   - Reach 100% coverage
   - Final testing and documentation

---

## Conclusion

This plan provides a structured approach to achieving 100% StableHLO operation support. The phased approach allows for incremental progress, regular testing, and early delivery of value. With dedicated effort over ~11 months, full support is achievable while maintaining TinyCC compatibility and non-GPL licensing requirements.
