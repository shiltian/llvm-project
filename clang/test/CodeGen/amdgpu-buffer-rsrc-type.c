
// RUN: %clang_cc1 -triple amdgcn -emit-llvm -o - %s -debug-info-kind=limited 2>&1 | FileCheck %s

// CHECK: name: "__buffer_rsrc_t",{{.*}}baseType: ![[BT:[0-9]+]]
// CHECK: [[BT]] = !DIBasicType(name: "__int128", size: 128, encoding: DW_ATE_signed)
void test_locals(void) {
  __buffer_rsrc_t k;
}
