# StableHLO to RISC-V ELF64 Code Generation

## Overview

The `stablehlo-codegen` tool compiles StableHLO MLIR programs to RISC-V ELF64 binaries.

## Prerequisites

1. **Rebuild LLVM with RISC-V support**: The build script has been updated to include RISC-V target. Rebuild MLIR:
   ```bash
   MLIR_ENABLE_BINDINGS_PYTHON=OFF build_tools/build_mlir.sh "${PWD}"/llvm-project/ "${PWD}"/llvm-build
   ```

2. **Rebuild StableHLO**: After LLVM rebuild completes, rebuild StableHLO:
   ```bash
   cd build
   cmake --build .
   ```

## Usage

### Basic Usage

```bash
stablehlo-codegen --input model.mlir --output model.o
```

### Command-Line Options

- `--input <file>`: Input MLIR file (required, or use `-` for stdin)
- `-o <file>`: Output file (default: stdout)
- `--target-triple <triple>`: Target triple (default: `riscv64-unknown-linux-gnu`)
- `--cpu <cpu>`: Target CPU (default: `generic-rv64`)
- `--features <features>`: Target features (default: `+m,+a,+f,+d`)
- `--O0`, `--O1`, `--O2`, `--O3`: Optimization level (default: `-O2`)
- `--emit-llvm`: Emit LLVM IR instead of object file
- `--emit-asm`: Emit assembly instead of object file

### Examples

```bash
# Compile to RISC-V object file
stablehlo-codegen --input test_riscv.mlir -o test_riscv.o

# Compile with specific CPU and features
stablehlo-codegen --input model.mlir -o model.o \
  --target-triple riscv64-unknown-linux-gnu \
  --cpu rocket-rv64 \
  --features "+m,+a,+f,+d,+c"

# Emit LLVM IR for inspection
stablehlo-codegen --input model.mlir -o model.ll --emit-llvm

# Emit assembly for inspection
stablehlo-codegen --input model.mlir -o model.s --emit-asm
```

## Compilation Pipeline

The tool performs the following transformations:

1. **StableHLO → Linalg**: Convert StableHLO operations to Linalg
2. **Bufferization**: Bufferize operations
3. **Linalg → Loops**: Lower Linalg to loops
4. **SCF → CF**: Convert structured control flow
5. **Affine → Standard**: Lower Affine operations
6. **Standard → LLVM Dialect**: Convert to LLVM dialect
7. **LLVM Dialect → LLVM IR**: Translate to LLVM IR
8. **LLVM IR → RISC-V**: Code generation to RISC-V assembly/object file

## Output Format

- **Object file** (default): ELF64 object file (`.o`) that can be linked
- **Assembly** (`--emit-asm`): RISC-V assembly (`.s`)
- **LLVM IR** (`--emit-llvm`): LLVM IR text (`.ll`)

## Notes

- The generated object files are position-independent and can be linked with a RISC-V linker
- Runtime functions (e.g., math operations) may need to be linked separately
- The tool supports RISC-V 64-bit (RV64) with standard extensions (M, A, F, D)
