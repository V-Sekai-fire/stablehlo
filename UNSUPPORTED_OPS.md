# Unsupported StableHLO Operations for C99 Codegen

## Summary

- **Total StableHLO Operations**: 115
- **Directly Supported**: 13 operations
- **Lowered to Supported Ops**: 10 operations
- **Lowered to CustomCall (C library functions)**: 15 operations
- **Total Supported/Lowered**: 38 operations (33%)
- **Unsupported**: 77 operations (67%)

## Directly Supported Operations (13)

1. `stablehlo.add` - Addition
2. `stablehlo.multiply` - Multiplication
3. `stablehlo.subtract` - Subtraction
4. `stablehlo.divide` - Division
5. `stablehlo.maximum` - Element-wise maximum
6. `stablehlo.remainder` - Modulo operation
7. `stablehlo.select` - Conditional selection
8. `stablehlo.and` - Logical AND
9. `stablehlo.not` - Logical NOT
10. `stablehlo.constant` - Constant values
11. `stablehlo.compare` - Comparison operations
12. `stablehlo.custom_call` - Custom function calls
13. `stablehlo.reduce` - Reduction (TODO: partial implementation)
14. `stablehlo.convolution` - Convolution (TODO: partial implementation)

## Lowered to Supported Operations (10)

1. `stablehlo.minimum` → `select` + `compare`
2. `stablehlo.or` → `!(!a && !b)` (De Morgan's law)
3. `stablehlo.xor` → Logical combination using `And`/`Not`
4. `stablehlo.negate` → `0 - x`
5. `stablehlo.abs` → `select(x >= 0, x, -x)`
6. `stablehlo.sign` → `select(x > 0, 1, select(x < 0, -1, 0))`
7. `stablehlo.rsqrt` → `1 / sqrt(x)`
8. `stablehlo.clamp` → `max(min(x, high), low)` → select chains
9. `stablehlo.logistic` → `1 / (1 + exp(-x))`
10. `stablehlo.expm1` → `exp(x) - 1`

## Lowered to CustomCall (C Library Functions) (15)

1. `stablehlo.sqrt` → `sqrt()`
2. `stablehlo.exponential` → `exp()`
3. `stablehlo.log` → `log()`
4. `stablehlo.floor` → `floor()`
5. `stablehlo.ceil` → `ceil()`
6. `stablehlo.round` → `round()`
7. `stablehlo.sine` → `sin()`
8. `stablehlo.cosine` → `cos()`
9. `stablehlo.tangent` → `tan()`
10. `stablehlo.tanh` → `tanh()`
11. `stablehlo.cbrt` → `cbrt()`
12. `stablehlo.log_plus_one` → `log1p()`
13. `stablehlo.is_finite` → `isfinite()`
14. `stablehlo.power` → `pow()`
15. `stablehlo.atan2` → `atan2()`

---

## Unsupported Operations (77)

### Shape/Tensor Manipulation (15)

1. `stablehlo.broadcast` - Broadcast tensor dimensions
2. `stablehlo.broadcast_in_dim` - Broadcast with dimension mapping
3. `stablehlo.dynamic_broadcast_in_dim` - Dynamic broadcast
4. `stablehlo.reshape` - Reshape tensor
5. `stablehlo.dynamic_reshape` - Dynamic reshape
6. `stablehlo.transpose` - Transpose dimensions
7. `stablehlo.reverse` - Reverse dimensions
8. `stablehlo.slice` - Slice tensor
9. `stablehlo.dynamic_slice` - Dynamic slice
10. `stablehlo.dynamic_update_slice` - Dynamic update slice
11. `stablehlo.real_dynamic_slice` - Real dynamic slice
12. `stablehlo.pad` - Pad tensor
13. `stablehlo.dynamic_pad` - Dynamic pad
14. `stablehlo.concatenate` - Concatenate tensors
15. `stablehlo.get_dimension_size` - Get dimension size

### Linear Algebra (7)

1. `stablehlo.dot` - Matrix multiplication
2. `stablehlo.dot_general` - General dot product
3. `stablehlo.einsum` - Einstein summation
4. `stablehlo.unary_einsum` - Unary einsum
5. `stablehlo.cholesky` - Cholesky decomposition
6. `stablehlo.triangular_solve` - Triangular solve
7. `stablehlo.reduce_window` - Reduce window operation

### Data Gathering/Scattering (6)

1. `stablehlo.gather` - Gather operation
2. `stablehlo.dynamic_gather` - Dynamic gather
3. `stablehlo.scatter` - Scatter operation
4. `stablehlo.select_and_scatter` - Select and scatter
5. `stablehlo.torch_index_select` - Torch-style index select
6. `stablehlo.reduce_scatter` - Reduce scatter

### Control Flow (4)

1. `stablehlo.if` - Conditional execution
2. `stablehlo.case` - Case/switch statement
3. `stablehlo.while` - While loop
4. `stablehlo.map` - Map operation

### Collective Operations (7)

1. `stablehlo.all_gather` - All-gather collective
2. `stablehlo.all_reduce` - All-reduce collective
3. `stablehlo.all_to_all` - All-to-all collective
4. `stablehlo.reduce_scatter` - Reduce-scatter collective
5. `stablehlo.collective_broadcast` - Collective broadcast
6. `stablehlo.collective_permute` - Collective permute
7. `stablehlo.cross_replica_sum` - Cross-replica sum

### Communication (4)

1. `stablehlo.send` - Send operation
2. `stablehlo.recv` - Receive operation
3. `stablehlo.infeed` - Infeed operation
4. `stablehlo.outfeed` - Outfeed operation

### Complex Numbers (2)

1. `stablehlo.complex` - Create complex number
2. `stablehlo.imag` - Imaginary part
3. `stablehlo.real` - Real part

### Type/Format Conversion (4)

1. `stablehlo.convert` - Type conversion
2. `stablehlo.bitcast_convert` - Bitcast conversion
3. `stablehlo.uniform_quantize` - Uniform quantization
4. `stablehlo.uniform_dequantize` - Uniform dequantization

### Special Operations (10)

1. `stablehlo.iota` - Generate iota sequence
2. `stablehlo.dynamic_iota` - Dynamic iota
3. `stablehlo.fft` - Fast Fourier Transform
4. `stablehlo.rng` - Random number generation
5. `stablehlo.rng_bit_generator` - RNG bit generator
6. `stablehlo.sort` - Sort operation
7. `stablehlo.reduce_precision` - Reduce precision
8. `stablehlo.optimization_barrier` - Optimization barrier
9. `stablehlo.composite` - Composite operation
10. `stablehlo.round_nearest_even` - Round to nearest even

### Batch Normalization (3)

1. `stablehlo.batch_norm_grad` - Batch norm gradient
2. `stablehlo.batch_norm_inference` - Batch norm inference
3. `stablehlo.batch_norm_training` - Batch norm training

### Dynamic Convolution (1)

1. `stablehlo.dynamic_conv` - Dynamic convolution

### Tuple Operations (2)

1. `stablehlo.tuple` - Create tuple
2. `stablehlo.get_tuple_element` - Get tuple element

### Token Operations (2)

1. `stablehlo.create_token` - Create token
2. `stablehlo.after_all` - After all tokens

### Bit Operations (3)

1. `stablehlo.shift_left` - Left shift
2. `stablehlo.shift_right_arithmetic` - Arithmetic right shift
3. `stablehlo.shift_right_logical` - Logical right shift
4. `stablehlo.clz` - Count leading zeros
5. `stablehlo.population_count` - Population count

### Dimension Operations (1)

1. `stablehlo.set_dimension_size` - Set dimension size

### Replica/Partition Operations (2)

1. `stablehlo.replica_id` - Get replica ID
2. `stablehlo.partition_id` - Get partition ID

### Return Operation (1)

1. `stablehlo.return` - Return operation (handled by func.return)

---

## Notes

### Operations with Partial/TODO Implementation

- `stablehlo.reduce` - Marked as supported but has TODO implementation
- `stablehlo.convolution` - Marked as supported but has TODO implementation

### Operations That May Be Lowerable

Many of these operations could potentially be lowered:

- **Shape operations**: Could potentially be handled via index manipulation
- **Type conversions**: Could use C casts
- **Bit operations**: Could use C bitwise operators
- **Simple reductions**: Could be lowered to loops
- **Control flow**: Could use C if/while statements

### Operations Requiring Special Handling

- **Collective operations**: Require distributed runtime support
- **Communication operations**: Require I/O infrastructure
- **Complex numbers**: Require complex number support
- **Quantization**: Require quantization-aware operations
- **Random number generation**: Require RNG library
- **FFT**: Require FFT library
