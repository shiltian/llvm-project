// RUN: %clang_cc1 -fclang-abi-compat=latest -triple amdgcn %s -emit-llvm -o - | FileCheck %s

namespace std { class type_info; };

auto &b = typeid(__buffer_rsrc_t);

// CHECK-DAG: @_ZTSu15__buffer_rsrc_t = {{.*}} c"u15__buffer_rsrc_t\00"
// CHECK-DAG: @_ZTIu15__buffer_rsrc_t = {{.*}} @_ZTVN10__cxxabiv123__fundamental_type_infoE, {{.*}} @_ZTSu15__buffer_rsrc_t
