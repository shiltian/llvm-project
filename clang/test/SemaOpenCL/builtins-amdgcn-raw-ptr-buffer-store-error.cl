// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu verde -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu tonga -S -verify -o - %s
// RUN: %clang_cc1 -triple amdgcn-unknown-unknown -target-cpu gfx1100 -S -verify -o - %s
// REQUIRES: amdgpu-registered-target

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef short v2i16 __attribute__((ext_vector_type(2)));
typedef int v2i32 __attribute__((ext_vector_type(2)));
typedef half v2f16 __attribute__((ext_vector_type(2)));
typedef float v2f32 __attribute__((ext_vector_type(2)));
typedef short v4i16 __attribute__((ext_vector_type(4)));
typedef int v4i32 __attribute__((ext_vector_type(4)));
typedef half v4f16 __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));

void test_amdgcn_raw_ptr_buffer_store_i8(char vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_i8(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_i8' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_i16(short vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_i16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_i16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_i32(int vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_i32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_i32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_f16(half vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_f16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_f16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_f32(float vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_f32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_f32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v2i16(v2i16 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v2i16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v2i16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v2i32(v2i32 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v2i32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v2i32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v2f16(v2f16 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v2f16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v2f16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v2f32(v2f32 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v2f32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v2f32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v4i16(v4i16 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v4i16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v4i16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v4i32(v4i32 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v4i32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v4i32' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v4f16(v4f16 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v4f16(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v4f16' must be a constant integer}}
}

void test_amdgcn_raw_ptr_buffer_store_v4f32(v4f32 vdata, __attribute__((address_space(8))) void *rsrc, int offset, int soffset, int aux) {
  __builtin_amdgcn_raw_ptr_buffer_store_v4f32(vdata, rsrc, /*offset=*/0, /*soffset=*/0, aux); //expected-error{{argument to '__builtin_amdgcn_raw_ptr_buffer_store_v4f32' must be a constant integer}}
}
