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
#include <cstdarg>  // For va_list
#include <cstdio>   // For vsnprintf
#include <cctype>   // For std::isspace

// Forward declaration
class String;

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
  explicit operator String() const { return String(str_); }
  // Stub for utf32 - implemented after String is defined
  struct Utf32View {
    const char32_t* ptr;
    size_t size;
    const char32_t* begin() const { return ptr; }
    const char32_t* end() const { return ptr + size; }
  };
  Utf32View utf32() const;
};

// Specialize std::hash for StringName
namespace std {
  template<>
  struct hash<StringName> {
    size_t operator()(const StringName& sn) const {
      return sn.hash();
    }
  };
}

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

// Vector - with write() and resize() methods (needed before String for String::split)
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
  T* ptrw() { return vec_.data(); }
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
  Vector<T> slice(size_t from) const {
    return slice(from, vec_.size());
  }
  typename std::vector<T>::iterator begin() { return vec_.begin(); }
  typename std::vector<T>::iterator end() { return vec_.end(); }
  typename std::vector<T>::const_iterator begin() const { return vec_.begin(); }
  typename std::vector<T>::const_iterator end() const { return vec_.end(); }
};

// String - alias for UString (Godot uses both) with additional methods
class String : public std::string {
public:
  String() : std::string() {}
  String(const char* s) : std::string(s) {}
  String(const std::string& s) : std::string(s) {}
  String(const StringName& sn) : std::string(sn.c_str()) {}
  String(size_t count, char ch) : std::string(count, ch) {}
  String& operator=(const std::string& s) { std::string::operator=(s); return *this; }
  String& operator+=(const String& s) { std::string::operator+=(s); return *this; }
  String& operator+=(const char* s) { std::string::operator+=(s); return *this; }
  String& operator+=(char c) { std::string::operator+=(c); return *this; }
  String operator+(const String& s) const { return String(std::string(*this) + std::string(s)); }
  String operator+(const char* s) const { return String(std::string(*this) + s); }
  String operator+(char c) const { return String(std::string(*this) + c); }
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
      result += String(1, (char)span[i]); // Simplified conversion
    }
    return result;
  }
  // Remove all occurrences of a character
  String remove_char(char ch) const {
    String result;
    for (char c : *this) {
      if (c != ch) {
        result.std::string::operator+=(c);
      }
    }
    return result;
  }
  // Static method to create String from char32_t
  static String chr(char32_t ch) {
    return String(1, (char)ch);
  }
  // Escape special characters for C strings
  String c_escape() const {
    String result;
    for (char c : *this) {
      switch (c) {
        case '\n': result.std::string::operator+=("\\n"); break;
        case '\t': result.std::string::operator+=("\\t"); break;
        case '\r': result.std::string::operator+=("\\r"); break;
        case '\\': result.std::string::operator+=("\\\\"); break;
        case '"': result.std::string::operator+=("\\\""); break;
        case '\'': result.std::string::operator+=("\\'"); break;
        default:
          if (c >= 32 && c < 127) {
            result.std::string::operator+=(c);
          } else {
            // Escape as octal
            char buf[5];
            snprintf(buf, sizeof(buf), "\\%03o", (unsigned char)c);
            result.std::string::operator+=(buf);
          }
          break;
      }
    }
    return result;
  }
  // Convert hex string to integer
  int64_t hex_to_int() const {
    return (int64_t)strtoll(c_str(), nullptr, 16);
  }
  // Convert binary string to integer
  int64_t bin_to_int() const {
    return (int64_t)strtoll(c_str(), nullptr, 2);
  }
  // Convert string to float
  double to_float() const {
    return strtod(c_str(), nullptr);
  }
  // Convert string to int
  int64_t to_int() const {
    return (int64_t)strtoll(c_str(), nullptr, 10);
  }
  // Convert string to int64 (alias for to_int)
  int64_t num_int64() const {
    return to_int();
  }
  // Convert string to uppercase
  String to_upper() const {
    String result = *this;
    for (size_t i = 0; i < result.length(); i++) {
      if (result[i] >= 'a' && result[i] <= 'z') {
        result[i] = result[i] - 'a' + 'A';
      }
    }
    return result;
  }
  // Capitalize first letter
  String capitalize() const {
    if (empty()) return String();
    String result = *this;
    if (result[0] >= 'a' && result[0] <= 'z') {
      result[0] = result[0] - 'a' + 'A';
    }
    return result;
  }
  // Escape XML special characters
  String xml_escape() const {
    String result;
    for (char c : *this) {
      switch (c) {
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '&': result += "&amp;"; break;
        case '"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        default: result += c; break;
      }
    }
    return result;
  }
  // Return hash value
  size_t hash() const {
    return std::hash<std::string>{}(*this);
  }
  // Split string by delimiter
  Vector<String> split(const String& delimiter, bool allow_empty = true, int maxsplit = 0) const {
    Vector<String> result;
    if (delimiter.empty()) {
      result.push_back(*this);
      return result;
    }
    size_t start = 0;
    size_t pos = 0;
    int count = 0;
    while ((pos = find(delimiter, start)) != std::string::npos) {
      if (allow_empty || pos > start) {
        result.push_back(String(substr(start, pos - start)));
        count++;
        if (maxsplit > 0 && count >= maxsplit) {
          result.push_back(String(substr(pos + delimiter.length())));
          return result;
        }
      }
      start = pos + delimiter.length();
    }
    if (allow_empty || start < length()) {
      result.push_back(String(substr(start)));
    }
    return result;
  }
  // Check if string is a valid integer
  bool is_valid_int() const {
    if (empty()) return false;
    const char* s = c_str();
    if (*s == '-' || *s == '+') s++;
    if (!*s) return false;
    while (*s) {
      if (*s < '0' || *s > '9') return false;
      s++;
    }
    return true;
  }
  // Quote the string
  String quote() const {
    return String("\"") + *this + "\"";
  }
  // Trim prefix from string
  String trim_prefix(const String& prefix) const {
    if (length() >= prefix.length() && substr(0, prefix.length()) == prefix) {
      return String(substr(prefix.length()));
    }
    return *this;
  }
  // Convert number to string (static)
  static String num_int64(int64_t num, int base = 10) {
    char buf[64];
    if (base == 16) {
      snprintf(buf, sizeof(buf), "%llx", (unsigned long long)num);
    } else {
      snprintf(buf, sizeof(buf), "%lld", (long long)num);
    }
    return String(buf);
  }
  // Check if string begins with prefix
  bool begins_with(const String& prefix) const {
    if (prefix.length() > length()) return false;
    return substr(0, prefix.length()) == prefix;
  }
  // Strip whitespace from edges
  String strip_edges() const {
    if (empty()) return String();
    size_t start = 0;
    size_t end = length();
    while (start < end && (std::isspace((*this)[start]) || (*this)[start] == '\t' || (*this)[start] == '\n' || (*this)[start] == '\r')) {
      start++;
    }
    while (end > start && (std::isspace((*this)[end-1]) || (*this)[end-1] == '\t' || (*this)[end-1] == '\n' || (*this)[end-1] == '\r')) {
      end--;
    }
    return String(substr(start, end - start));
  }
};

// ObjectID - stub
using ObjectID = uint64_t;

// Ref - template for reference counting (works with forward declarations via helper functions)
template<typename T>
class Ref {
  T* ptr_;
public:
  Ref() : ptr_(nullptr) {}
  Ref(T* p) : ptr_(p) { if (ptr_) reference_helper(ptr_); }
  ~Ref() { if (ptr_) unreference_helper(ptr_); }
  Ref(const Ref& other) : ptr_(other.ptr_) { if (ptr_) reference_helper(ptr_); }
  Ref& operator=(const Ref& other) {
    if (other.ptr_) reference_helper(other.ptr_);
    if (ptr_) unreference_helper(ptr_);
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

// Helper functions for reference counting (work with incomplete types via casting)
template<typename T>
void reference_helper(T* p) {
  if (p) {
    RefCounted* rc = static_cast<RefCounted*>(p);
    rc->reference();
  }
}

template<typename T>
void unreference_helper(T* p) {
  if (p) {
    RefCounted* rc = static_cast<RefCounted*>(p);
    rc->unreference();
  }
}

// GDScript - forward declaration stub (defined after Resource)
class GDScript;


// PropertyInfo - stub
struct PropertyInfo {
  StringName name;
  StringName type_name;
  int hint = 0;
  String hint_string;
};

// PropertyHint - stub enum
enum PropertyHint {
  PROPERTY_HINT_NONE = 0
};

// PropertyUsageFlags - stub enum
enum PropertyUsageFlags {
  PROPERTY_USAGE_NONE = 0
};

// Object - stub base class
class Object : public RefCounted {};

// NodePath - stub
class NodePath {
  std::string path_;
public:
  NodePath() {}
  NodePath(const String& s) : path_(s) {}
  NodePath(const char* s) : path_(s) {}
  const char* c_str() const { return path_.c_str(); }
  operator String() const { return String(path_); }
};

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
  GuestVariant(const String& s) : type_(STRING), string_val(new std::string(s)) {}
  GuestVariant(const StringName& sn) : type_(STRING), string_val(new std::string(sn.c_str())) {}
  
  GuestVariant(const GuestVariant& other) : type_(other.type_) {
    if (type_ == STRING && other.string_val) {
      string_val = new std::string(*other.string_val);
    } else {
      bool_val = other.bool_val;
      int_val = other.int_val;
      float_val = other.float_val;
      object_val = other.object_val;
    }
  }
  
  GuestVariant& operator=(const GuestVariant& other) {
    if (this == &other) return *this;
    if (type_ == STRING && string_val) {
      delete string_val;
    }
    type_ = other.type_;
    if (type_ == STRING && other.string_val) {
      string_val = new std::string(*other.string_val);
    } else {
      bool_val = other.bool_val;
      int_val = other.int_val;
      float_val = other.float_val;
      object_val = other.object_val;
    }
    return *this;
  }
  
  GuestVariant& operator=(const StringName& sn) {
    if (type_ == STRING && string_val) {
      delete string_val;
    }
    type_ = STRING;
    string_val = new std::string(sn.c_str());
    return *this;
  }
  
  GuestVariant& operator=(const String& s) {
    if (type_ == STRING && string_val) {
      delete string_val;
    }
    type_ = STRING;
    string_val = new std::string(s);
    return *this;
  }
  
  GuestVariant& operator=(const char* s) {
    if (type_ == STRING && string_val) {
      delete string_val;
    }
    type_ = STRING;
    string_val = new std::string(s);
    return *this;
  }
  
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
  
  // Operator enum for variant operations
  enum Operator {
    OP_EQUAL,
    OP_NOT_EQUAL,
    OP_LESS,
    OP_LESS_EQUAL,
    OP_GREATER,
    OP_GREATER_EQUAL,
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_MODULE,
    OP_POWER,
    OP_SHIFT_LEFT,
    OP_SHIFT_RIGHT,
    OP_BIT_AND,
    OP_BIT_OR,
    OP_BIT_XOR,
    OP_BIT_NEGATE,
    OP_AND,
    OP_OR,
    OP_XOR,
    OP_NOT,
    OP_IN,
    OP_MAX
  };
  
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

// KeyValue helper for HashMap iteration (must be before HashMap)
template<typename K, typename V>
struct KeyValue {
  K key;
  V value;
  KeyValue() {}
  KeyValue(const K& k, const V& v) : key(k), value(v) {}
  KeyValue(const std::pair<const K, V>& p) : key(p.first), value(p.second) {}
};

// HashMap - simplified with iterator support
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
  void erase(const K& key) { map_.erase(key); }
  
  // Iterator that returns KeyValue for range-based for loops
  class iterator {
    typename std::unordered_map<K, V>::iterator it_;
  public:
    iterator(typename std::unordered_map<K, V>::iterator it) : it_(it) {}
    KeyValue<K, V> operator*() { return KeyValue<K, V>(*it_); }
    iterator& operator++() { ++it_; return *this; }
    bool operator!=(const iterator& other) const { return it_ != other.it_; }
  };
  
  class const_iterator {
    typename std::unordered_map<K, V>::const_iterator it_;
  public:
    const_iterator(typename std::unordered_map<K, V>::const_iterator it) : it_(it) {}
    KeyValue<K, V> operator*() { return KeyValue<K, V>(*it_); }
    const_iterator& operator++() { ++it_; return *this; }
    bool operator!=(const const_iterator& other) const { return it_ != other.it_; }
  };
  
  iterator begin() { return iterator(map_.begin()); }
  iterator end() { return iterator(map_.end()); }
  const_iterator begin() const { return const_iterator(map_.begin()); }
  const_iterator end() const { return const_iterator(map_.end()); }
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


// ScriptLanguage - stub
class ScriptLanguage {};

// MethodInfo - stub
struct MethodInfo {
  StringName name;
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
#define ERR_FAIL_COND_MSG(cond, msg) if (cond) return
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
  static String path_remap(const String& path) {
    // Stub - just return the path as-is
    return path;
  }
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
  va_list args;
  va_start(args, fmt);
  char buffer[4096];
  vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);
  return String(buffer);
}

// Helper to convert String to const char* for variadic functions
inline const char* string_to_cstr(const String& s) {
  return s.c_str();
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

inline Error decode_variant(GuestVariant& r_v, const uint8_t* p_buffer, int p_len, int* r_len, bool p_allow_objects = false) {
  // Simplified stub - just return OK for now
  if (r_len) *r_len = 0;
  return OK;
}

// GDScriptCache stub
class GDScriptCache {
public:
  static Vector<uint8_t> get_binary_tokens(const String& path) {
    return Vector<uint8_t>(); // Stub
  }
  static String get_source_code(const String& path) {
    return String(); // Stub
  }
  static GDScriptCache* singleton;
  struct MutexLock {
    MutexLock(void* mutex) {}
  };
  void* mutex = nullptr;
  HashMap<String, void*> parser_map;
};

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
