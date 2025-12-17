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

// Minimal adapters for Godot core types to allow GDScriptParser to compile standalone
// These are simplified versions that provide the minimal interface needed

#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <list>
#include <memory>
#include <cstdint>
#include <cstring>  // For memcpy

// StringName - simplified version
class StringName {
  std::string str_;
public:
  StringName() {}
  StringName(const char* s) : str_(s) {}
  StringName(const std::string& s) : str_(s) {}
  const std::string& operator*() const { return str_; }
  const char* c_str() const { return str_.c_str(); }
  bool operator==(const StringName& other) const { return str_ == other.str_; }
  bool operator!=(const StringName& other) const { return str_ != other.str_; }
  size_t hash() const { return std::hash<std::string>{}(str_); }
};

// UString - simplified (just std::string for now)
using UString = std::string;

// Span - simplified view type (needed before String)
template<typename T>
class Span {
  const T* data_;
  size_t size_;
public:
  Span(const T* data, size_t size) : data_(data), size_(size) {}
  const T* ptr() const { return data_; }
  size_t size() const { return size_; }
  const T& operator[](size_t i) const { return data_[i]; }
};

// String - alias for UString (Godot uses both) with additional methods
class String : public std::string {
public:
  String() : std::string() {}
  String(const char* s) : std::string(s) {}
  String(const std::string& s) : std::string(s) {}
  String(const StringName& sn) : std::string(sn.c_str()) {}
  String& operator=(const std::string& s) { std::string::operator=(s); return *this; }
  String& operator+=(const String& s) { std::string::operator+=(s); return *this; }
  String operator+(const String& s) const { return String(std::string(*this) + std::string(s)); }
  const char* get_data() const { return c_str(); }
  char* get_data() { return data(); }
  // Stub for utf32 - returns a span-like structure
  struct Utf32View {
    const char32_t* ptr;
    size_t size;
    const char32_t* begin() const { return ptr; }
    const char32_t* end() const { return ptr + size; }
  };
  Utf32View utf32() const {
    // Simplified - convert to char32_t array
    static char32_t buf[1024];
    size_t i = 0;
    for (char c : *this) {
      if (i < 1023) buf[i++] = (char32_t)(unsigned char)c;
    }
    buf[i] = 0;
    return {buf, i};
  }
  // Static version that takes a Span
  static String utf32(Span<const char32_t> span) {
    String result;
    for (size_t i = 0; i < span.size(); i++) {
      result += (char)span[i]; // Simplified conversion
    }
    return result;
  }
};

// ObjectID - stub
using ObjectID = uint64_t;

// Ref - template for reference counting
template<typename T>
class Ref {
  T* ptr_;
public:
  Ref() : ptr_(nullptr) {}
  Ref(T* p) : ptr_(p) { if (ptr_) ptr_->reference(); }
  ~Ref() { if (ptr_) ptr_->unreference(); }
  Ref(const Ref& other) : ptr_(other.ptr_) { if (ptr_) ptr_->reference(); }
  Ref& operator=(const Ref& other) {
    if (other.ptr_) other.ptr_->reference();
    if (ptr_) ptr_->unreference();
    ptr_ = other.ptr_;
    return *this;
  }
  T* operator->() { return ptr_; }
  const T* operator->() const { return ptr_; }
  T* ptr() { return ptr_; }
  const T* ptr() const { return ptr_; }
  bool is_valid() const { return ptr_ != nullptr; }
  bool is_null() const { return ptr_ == nullptr; }
};

// GDScript - forward declaration stub (defined after Resource)
class GDScript;

// GuestVariant - minimal stub (used instead of full Variant for standalone parser)
class GuestVariant {
public:
  enum Type {
    NIL,
    BOOL,
    INT,
    FLOAT,
    STRING,
    OBJECT,
    VARIANT_MAX
  };
  
  Type type_;
  union {
    bool bool_val;
    int64_t int_val;
    double float_val;
    std::string* string_val;
    void* object_val;
  };
  
  GuestVariant() : type_(NIL) {}
  GuestVariant(bool b) : type_(BOOL), bool_val(b) {}
  GuestVariant(int64_t i) : type_(INT), int_val(i) {}
  GuestVariant(double f) : type_(FLOAT), float_val(f) {}
  GuestVariant(const std::string& s) : type_(STRING), string_val(new std::string(s)) {}
  
  ~GuestVariant() {
    if (type_ == STRING && string_val) {
      delete string_val;
    }
  }
  
  Type get_type() const { return type_; }
  static const char* get_type_name(Type t) {
    switch (t) {
      case NIL: return "Nil";
      case BOOL: return "Bool";
      case INT: return "Int";
      case FLOAT: return "Float";
      case STRING: return "String";
      case OBJECT: return "Object";
      default: return "Unknown";
    }
  }
  
  bool booleanize() const {
    if (type_ == BOOL) return bool_val;
    if (type_ == INT) return int_val != 0;
    if (type_ == FLOAT) return float_val != 0.0;
    return false;
  }
  
  operator int64_t() const {
    if (type_ == INT) return int_val;
    if (type_ == FLOAT) return (int64_t)float_val;
    return 0;
  }
  
  operator double() const {
    if (type_ == FLOAT) return float_val;
    if (type_ == INT) return (double)int_val;
    return 0.0;
  }
  
  operator std::string() const {
    if (type_ == STRING && string_val) return *string_val;
    return "";
  }
};

// Variant - alias to GuestVariant for compatibility (must be after GuestVariant definition)
using Variant = GuestVariant;

// HashMap - simplified
template<typename K, typename V>
class HashMap {
  std::unordered_map<K, V> map_;
public:
  bool has(const K& key) const { return map_.find(key) != map_.end(); }
  V& operator[](const K& key) { return map_[key]; }
  const V& operator[](const K& key) const { return map_.at(key); }
  void clear() { map_.clear(); }
  bool is_empty() const { return map_.empty(); }
  size_t size() const { return map_.size(); }
};

// Vector - with write() and resize() methods
template<typename T>
class Vector {
  std::vector<T> vec_;
public:
  void push_back(const T& val) { vec_.push_back(val); }
  void resize(size_t size) { vec_.resize(size); }
  void resize(size_t size, const T& val) { vec_.resize(size, val); }
  T& write(size_t i) { 
    if (i >= vec_.size()) vec_.resize(i + 1);
    return vec_[i]; 
  }
  T& write(size_t i, const T& val) {
    if (i >= vec_.size()) vec_.resize(i + 1);
    vec_[i] = val;
    return vec_[i];
  }
  const T& operator[](size_t i) const { return vec_[i]; }
  T& operator[](size_t i) { return vec_[i]; }
  T* ptr() { return vec_.data(); }
  const T* ptr() const { return vec_.data(); }
  size_t size() const { return vec_.size(); }
  void clear() { vec_.clear(); }
  bool is_empty() const { return vec_.empty(); }
  void append_array(const Vector<T>& other) {
    vec_.insert(vec_.end(), other.vec_.begin(), other.vec_.end());
  }
  Vector<T> slice(size_t from, size_t to) const {
    Vector<T> result;
    if (from < vec_.size() && to <= vec_.size() && from <= to) {
      result.vec_.assign(vec_.begin() + from, vec_.begin() + to);
    }
    return result;
  }
  T* ptrw() { return vec_.data(); }
  const T* ptr() const { return vec_.data(); }
};

// List - simplified
template<typename T>
class List {
  std::list<T> list_;
public:
  void push_back(const T& val) { list_.push_back(val); }
  void pop_back() { list_.pop_back(); }
  T& back() { return list_.back(); }
  const T& back() const { return list_.back(); }
  T& get(size_t idx) {
    auto it = list_.begin();
    std::advance(it, idx);
    return *it;
  }
  const T& get(size_t idx) const {
    auto it = list_.begin();
    std::advance(it, idx);
    return *it;
  }
  size_t size() const { return list_.size(); }
  void clear() { list_.clear(); }
  bool is_empty() const { return list_.empty(); }
  typename std::list<T>::iterator begin() { return list_.begin(); }
  typename std::list<T>::iterator end() { return list_.end(); }
  typename std::list<T>::const_iterator begin() const { return list_.begin(); }
  typename std::list<T>::const_iterator end() const { return list_.end(); }
};

// RefCounted - stub
class RefCounted {
public:
  void reference() {}
  void unreference() {}
};

// Resource - stub
class Resource : public RefCounted {};

// ScriptLanguage - stub
class ScriptLanguage {};

// MethodInfo - stub
struct MethodInfo {
  StringName name;
};

// KeyValue - helper for HashMap iteration
template<typename K, typename V>
struct KeyValue {
  K key;
  V value;
};

// Error codes
enum Error {
  OK = 0,
  ERR_FILE_NOT_FOUND,
  ERR_INVALID_DATA,
  ERR_INVALID_PARAMETER
};

// Global defines that might be needed
#define ERR_FAIL_V(expr, msg) return expr
#define ERR_FAIL_V_MSG(expr, msg) return expr
#define ERR_FAIL_INDEX(idx, size) (void)0
#define ERR_FAIL_INDEX_V(idx, size, expr) if ((idx) >= (size)) return expr
#define ERR_FAIL_INDEX_V_MSG(idx, size, expr, msg) if ((idx) >= (size)) return expr
#define ERR_FAIL_COND(cond) if (cond) return
#define ERR_FAIL_COND_V(cond, expr) if (cond) return expr
#define ERR_FAIL_COND_V_MSG(cond, expr, msg) if (cond) return expr
#define unlikely(x) (x)
// std_size macro - works with both containers and arrays
template<typename T, size_t N>
constexpr size_t std_size(const T (&)[N]) { return N; }
template<typename T>
constexpr size_t std_size(const T& container) { return container.size(); }
#define std_size(x) std_size(x)

// Project settings stubs
#define GLOBAL_GET(path) Variant(false)

// Editor settings stubs
#ifdef TOOLS_ENABLED
namespace EditorSettings {
  bool get_setting(const StringName& name, bool default_val = false) { return default_val; }
}
#endif

// ResourceLoader stub
class ResourceLoader {
public:
  static Resource* load(const StringName& path, const StringName& type_hint = StringName()) { return nullptr; }
};

// ProjectSettings stub
class ProjectSettings {
public:
  static bool has_setting(const StringName& name) { return false; }
  static GuestVariant get_setting(const StringName& name) { return GuestVariant(); }
};

// MultiplayerAPI stub
class MultiplayerAPI {
public:
  enum RPCMode {
    RPC_MODE_DISABLED
  };
};

// TextServer stub (for DEBUG_ENABLED)
#ifdef DEBUG_ENABLED
class TextServer {
public:
  static int get_singleton() { return 0; }
};
#endif

// Memory management stubs
template<typename T>
T* memnew(T* obj = nullptr) {
  if (obj == nullptr) {
    return new T();
  }
  return new T(*obj);
}

template<typename T>
void memdelete(T* obj) {
  delete obj;
}

// Additional stubs that might be needed
// Note: GDScriptParser is a class defined in gdscript_parser.h
// We use a forward declaration here, but the actual definition must be included
// where GDScriptAnalyzer is used
class GDScriptParser;
class GDScriptAnalyzer {
public:
  GDScriptAnalyzer(GDScriptParser* parser) {}
  Error analyze() { return OK; }
};

// String helpers
inline String vformat(const char* fmt, ...) {
  return String(fmt);  // Simplified
}

// Marshalling functions
inline uint32_t decode_uint32(const uint8_t* p_r) {
  return (uint32_t)p_r[0] | ((uint32_t)p_r[1] << 8) | 
         ((uint32_t)p_r[2] << 16) | ((uint32_t)p_r[3] << 24);
}

inline void encode_uint32(uint32_t p_val, uint8_t* p_dest) {
  p_dest[0] = (uint8_t)(p_val & 0xFF);
  p_dest[1] = (uint8_t)((p_val >> 8) & 0xFF);
  p_dest[2] = (uint8_t)((p_val >> 16) & 0xFF);
  p_dest[3] = (uint8_t)((p_val >> 24) & 0xFF);
}

// GuestVariant encoding/decoding stubs
inline int encode_variant(const GuestVariant& p_val, uint8_t* p_buffer, int p_len, bool p_allow_objects = false) {
  // Simplified stub - just return 0 for now
  // Full implementation would serialize the variant
  return 0;
}

inline int decode_variant(GuestVariant& r_v, const uint8_t* p_buffer, int p_len, bool p_allow_objects = false) {
  // Simplified stub - just return 0 for now
  return 0;
}

// Compression stubs
class Compression {
public:
  enum Mode {
    MODE_ZSTD
  };
  static int64_t get_max_compressed_buffer_size(int64_t p_uncompressed_size, Mode p_mode) {
    return p_uncompressed_size * 2; // Simplified
  }
  static int compress(uint8_t* p_dst, const uint8_t* p_src, int p_src_size, Mode p_mode) {
    // Stub - no compression
    memcpy(p_dst, p_src, p_src_size);
    return p_src_size;
  }
  static int decompress(uint8_t* p_dst, int p_dst_max_size, const uint8_t* p_src, int p_src_size, Mode p_mode) {
    // Stub - no decompression
    int size = (p_src_size < p_dst_max_size) ? p_src_size : p_dst_max_size;
    memcpy(p_dst, p_src, size);
    return size;
  }
};
