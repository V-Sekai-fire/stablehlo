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

#ifndef STABLEHLO_TOOLS_GDSCRIPTSYSCALLMAP_H_
#define STABLEHLO_TOOLS_GDSCRIPTSYSCALLMAP_H_

#include "llvm/ADT/StringRef.h"
#include <string>
#include <unordered_map>

namespace mlir {
namespace stablehlo {

/// Maps GDScript operations to syscall names based on sandbox module definitions
class GDScriptSyscallMap {
public:
  /// Check if a function call should be a syscall
  static bool isSyscall(llvm::StringRef functionName);
  
  /// Get syscall name for a function (e.g., "print" -> "godot_syscalls_print")
  static std::string getSyscallName(llvm::StringRef functionName);
  
  /// Check if an operation is a Godot syscall (vs Linux syscall)
  static bool isGodotSyscall(llvm::StringRef functionName);
  
  /// Get the base syscall name without prefix
  static std::string getBaseSyscallName(llvm::StringRef functionName);
  
  /// Initialize syscall mappings (called once at startup)
  static void initialize();

private:
  static std::unordered_map<std::string, std::string> syscallMap;
  static bool initialized;
  
  /// Map common GDScript functions to syscall names
  static void populateSyscallMap();
};

} // namespace stablehlo
} // namespace mlir

#endif // STABLEHLO_TOOLS_GDSCRIPTSYSCALLMAP_H_
