# Analysis: Dropping C99 Requirement

## Current State

- **Target**: C99 codegen
- **Compiler**: TinyCC (C99 compatible)
- **Libraries**: SIMDe (C99), SLEEF (C99/C11)
- **Limitations**: Cannot use C++ libraries (xsimd, xtensor)

## Options if We Drop C99

### Option 1: C11/C17 (Still C, Not C++)

**Benefits**:
- ✅ Still C (not C++)
- ✅ More features than C99 (generic macros, better Unicode, etc.)
- ✅ SIMDe/SLEEF still work
- ✅ Can use more modern C features

**Limitations**:
- ❌ Still can't use C++ libraries (xsimd, xtensor)
- ❌ TinyCC has limited C11 support
- ⚠️ May need different compiler (GCC, Clang)

**Impact on Plan**:
- Minimal changes needed
- Still use SIMDe/SLEEF approach
- Same implementation strategy

### Option 2: C++11/C++14 (C++ but older standard)

**Benefits**:
- ✅ Can use xsimd (requires C++14)
- ✅ Can use some C++ features (templates, classes)
- ✅ Better type safety
- ✅ More libraries available

**Limitations**:
- ❌ Cannot use xtensor (requires C++17+)
- ❌ TinyCC doesn't support C++
- ⚠️ Need C++ compiler (GCC, Clang)
- ⚠️ Generated code is C++, not C

**Impact on Plan**:
- Can use xsimd for SIMD operations
- Still need manual tensor operations
- Different codegen approach

### Option 3: C++17/C++20 (Modern C++)

**Benefits**:
- ✅ Can use xsimd (C++14+)
- ✅ Can use xtensor (C++17+, some features need C++20)
- ✅ Full tensor library support
- ✅ Better abstractions

**Limitations**:
- ❌ TinyCC doesn't support C++
- ⚠️ Need modern C++ compiler
- ⚠️ Generated code is C++, not C
- ⚠️ Larger binary size

**Impact on Plan**:
- Can use xsimd + xtensor (as originally planned)
- Much easier implementation for tensor operations
- Better performance with xtensor
- Follows original xsimd/xtensor analysis

### Option 4: C11 with C++ Compiler (Hybrid)

**Benefits**:
- ✅ Use C11 features
- ✅ Can link against C++ libraries (if pre-compiled)
- ✅ Generated code is still C

**Limitations**:
- ❌ Cannot use C++ features in generated code
- ❌ Still need manual tensor operations
- ⚠️ Complex linking setup

**Impact on Plan**:
- Similar to C11 option
- Could link against pre-compiled C++ libraries

## Recommendation Matrix

| Option | TinyCC Compatible | xsimd | xtensor | Ease of Implementation | Performance |
|--------|------------------|-------|---------|------------------------|-------------|
| **C99** (current) | ✅ Yes | ❌ No | ❌ No | Medium | Good (with SIMDe) |
| **C11** | ⚠️ Partial | ❌ No | ❌ No | Medium | Good (with SIMDe) |
| **C++11/14** | ❌ No | ✅ Yes | ❌ No | Medium | Better (with xsimd) |
| **C++17/20** | ❌ No | ✅ Yes | ✅ Yes | **Easier** | **Best** |

## What Changes if We Drop C99?

### If We Choose C++17/C++20 (Recommended)

**Code Changes**:
1. Update `CCodegen.cpp` → `CppCodegen.cpp`
2. Change header from C99 to C++17/20
3. Use C++ features (templates, classes, etc.)
4. Integrate xsimd and xtensor

**Plan Changes**:
- **Phase 1**: Use xsimd instead of SIMDe (better C++ integration)
- **Phase 2**: Use SLEEF C++ API or keep C API
- **Phase 3**: Use xtensor for shape operations (much easier!)
- **Phase 4**: Use xtensor for linear algebra (easier!)

**Coverage Improvement**:
- Same 100% target
- **Easier implementation** for tensor operations
- **Better performance** with xtensor optimizations
- **Faster development** (weeks instead of months for tensor ops)

**Compiler Requirements**:
- Need C++17/C++20 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Cannot use TinyCC (no C++ support)
- Can still target RISC-V with `riscv64-unknown-linux-gnu-g++`

### If We Choose C11 (Minimal Change)

**Code Changes**:
1. Update comments from "C99" to "C11"
2. Use C11 features (generic macros, etc.)
3. Keep SIMDe/SLEEF approach

**Plan Changes**:
- Minimal changes
- Still use SIMDe/SLEEF
- Still manual tensor operations
- Similar timeline

**Compiler Requirements**:
- Need C11 compiler (GCC 4.7+, Clang 3.1+)
- TinyCC has limited C11 support
- May need GCC/Clang

## Recommendation

### **Option: C++17/C++20** (Best Overall)

**Why**:
1. **Easier Implementation**: xtensor handles tensor operations automatically
2. **Better Performance**: Optimized tensor libraries
3. **Faster Development**: Weeks instead of months for tensor ops
4. **More Libraries**: Access to entire C++ ecosystem
5. **Future-Proof**: Modern standard with active development

**Trade-offs**:
- ❌ Cannot use TinyCC (need C++ compiler)
- ⚠️ Generated code is C++, not C
- ⚠️ Larger binary size (but acceptable)

**Updated Timeline**:
- **Original (C99)**: 44 weeks
- **With C++17/20**: ~30-35 weeks (faster due to xtensor)

### Alternative: Keep C99 but Drop TinyCC Requirement

If the goal is just to drop C99 (not necessarily keep TinyCC):

**Option**: C11 with GCC/Clang
- Still C (not C++)
- More features than C99
- Better compiler support
- Can still use SIMDe/SLEEF

## Questions to Answer

1. **What's the actual target?**
   - TinyCC specifically?
   - Any C compiler?
   - Any C++ compiler?
   - RISC-V target specifically?

2. **What's the use case?**
   - Embedded systems (may need C)?
   - General purpose (C++ OK)?
   - Performance critical (C++ better)?

3. **What's the deployment environment?**
   - Can users install C++ compilers?
   - Need minimal dependencies?
   - Binary size constraints?

## Next Steps

Once we decide on the target language/standard:

1. **Update codegen**: Change from C99 to chosen standard
2. **Update plan**: Adjust implementation strategy
3. **Update libraries**: Choose appropriate libraries
4. **Update tests**: Test with appropriate compiler

