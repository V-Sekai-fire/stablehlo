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

#include "stablehlo/tools/GDScriptSyscallMap.h"
#include <algorithm>
#include <cctype>

namespace mlir {
namespace stablehlo {

std::unordered_map<std::string, std::string> GDScriptSyscallMap::syscallMap;
bool GDScriptSyscallMap::initialized = false;

void GDScriptSyscallMap::initialize() {
  if (initialized) return;
  
  populateSyscallMap();
  initialized = true;
}

void GDScriptSyscallMap::populateSyscallMap() {
  // Map based on ECALL_* definitions from godot/modules/sandbox/src/syscalls.h
  // GAME_API_BASE = 500
  
  // ECALL_PRINT (GAME_API_BASE + 0)
  syscallMap["print"] = "godot_syscalls_print";
  syscallMap["printl"] = "godot_syscalls_print";
  
  // ECALL_VCALL (GAME_API_BASE + 1) - Variant call
  // ECALL_VEVAL (GAME_API_BASE + 2) - Variant eval
  // ECALL_VASSIGN (GAME_API_BASE + 3) - Variant assign
  
  // ECALL_GET_OBJ (GAME_API_BASE + 4) - Get object by name
  syscallMap["get_node"] = "godot_syscalls_get_node";
  
  // ECALL_OBJ (GAME_API_BASE + 5) - Object functions
  // ECALL_OBJ_CALLP (GAME_API_BASE + 6) - Call method on object
  syscallMap["call"] = "godot_syscalls_obj_callp";
  syscallMap["call_deferred"] = "godot_syscalls_obj_callp";
  
  // ECALL_GET_NODE (GAME_API_BASE + 7) - Get node by path
  syscallMap["get_node"] = "godot_syscalls_get_node";
  
  // ECALL_NODE (GAME_API_BASE + 8) - Node functions
  syscallMap["queue_free"] = "godot_syscalls_node_queue_free";
  syscallMap["get_name"] = "godot_syscalls_node_get_name";
  syscallMap["set_name"] = "godot_syscalls_node_set_name";
  
  // ECALL_NODE2D (GAME_API_BASE + 9) - Node2D functions
  syscallMap["get_position"] = "godot_syscalls_node2d_get_position";
  syscallMap["set_position"] = "godot_syscalls_node2d_set_position";
  syscallMap["get_rotation"] = "godot_syscalls_node2d_get_rotation";
  syscallMap["set_rotation"] = "godot_syscalls_node2d_set_rotation";
  
  // ECALL_NODE3D (GAME_API_BASE + 10) - Node3D functions
  syscallMap["get_transform"] = "godot_syscalls_node3d_get_transform";
  syscallMap["set_transform"] = "godot_syscalls_node3d_set_transform";
  
  // ECALL_THROW (GAME_API_BASE + 11)
  syscallMap["push_error"] = "godot_syscalls_throw";
  syscallMap["assert"] = "godot_syscalls_throw";
  
  // ECALL_IS_EDITOR (GAME_API_BASE + 12)
  syscallMap["is_editor_hint"] = "godot_syscalls_is_editor";
  
  // Math operations (ECALL_MATH_OP32, ECALL_MATH_OP64)
  syscallMap["sin"] = "godot_syscalls_math_sin";
  syscallMap["cos"] = "godot_syscalls_math_cos";
  syscallMap["tan"] = "godot_syscalls_math_tan";
  syscallMap["asin"] = "godot_syscalls_math_asin";
  syscallMap["acos"] = "godot_syscalls_math_acos";
  syscallMap["atan"] = "godot_syscalls_math_atan";
  syscallMap["atan2"] = "godot_syscalls_math_atan2";
  syscallMap["pow"] = "godot_syscalls_math_pow";
  
  // Vector operations
  syscallMap["length"] = "godot_syscalls_vec2_length";
  syscallMap["normalized"] = "godot_syscalls_vec2_normalized";
  syscallMap["rotated"] = "godot_syscalls_vec2_rotated";
  
  // Array operations (ECALL_ARRAY_OPS, etc.)
  syscallMap["size"] = "godot_syscalls_array_size";
  syscallMap["push_back"] = "godot_syscalls_array_push_back";
  syscallMap["pop_back"] = "godot_syscalls_array_pop_back";
  
  // String operations (ECALL_STRING_OPS, etc.)
  syscallMap["str"] = "godot_syscalls_string_create";
  syscallMap["to_string"] = "godot_syscalls_string_create";
  
  // Object property access
  syscallMap["get"] = "godot_syscalls_obj_prop_get";
  syscallMap["set"] = "godot_syscalls_obj_prop_set";
}

bool GDScriptSyscallMap::isSyscall(llvm::StringRef functionName) {
  initialize();
  
  std::string name = functionName.str();
  // Convert to lowercase for case-insensitive matching
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  
  return syscallMap.find(name) != syscallMap.end();
}

std::string GDScriptSyscallMap::getSyscallName(llvm::StringRef functionName) {
  initialize();
  
  std::string name = functionName.str();
  // Convert to lowercase for case-insensitive matching
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);
  
  auto it = syscallMap.find(name);
  if (it != syscallMap.end()) {
    return it->second;
  }
  
  // Default: assume it's a Godot syscall with the function name
  return "godot_syscalls_" + name;
}

bool GDScriptSyscallMap::isGodotSyscall(llvm::StringRef functionName) {
  // For now, all syscalls are Godot syscalls
  // Linux syscalls would be things like file I/O, network, etc.
  // Those can be added later if needed
  return isSyscall(functionName);
}

std::string GDScriptSyscallMap::getBaseSyscallName(llvm::StringRef functionName) {
  std::string syscallName = getSyscallName(functionName);
  // Remove "godot_syscalls_" or "linux_syscalls_" prefix
  if (syscallName.find("godot_syscalls_") == 0) {
    return syscallName.substr(15); // length of "godot_syscalls_"
  } else if (syscallName.find("linux_syscalls_") == 0) {
    return syscallName.substr(15); // length of "linux_syscalls_"
  }
  return syscallName;
}

} // namespace stablehlo
} // namespace mlir
