//===------- BuiltinAllocator.cpp - Generic GPU memory allocator -- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#pragma omp begin declare target device_type(nohost)

#include "Debug.h"
#include "Memory.h"
#include "Utils.h"

using namespace ompx;

namespace ompx {
namespace memory {
MemoryAllocationInfo getMemoryAllocationInfo(void *P) { return {}; }

void init() {}
} // namespace memory
} // namespace ompx

extern "C" {

void *realloc(void *ptr, size_t new_size) {
  void *NewPtr = malloc(new_size);
  if (!NewPtr)
    return nullptr;
  __builtin_memcpy(NewPtr, ptr, new_size);
  free(ptr);
  return NewPtr;
}
}

#pragma omp end declare target
