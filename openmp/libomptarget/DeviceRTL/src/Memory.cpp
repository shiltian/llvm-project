//===------- Memory.cpp - OpenMP device runtime memory allocator -- C++ -*-===//
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

#include "Memory.h"
#include "Synchronization.h"

using namespace _OMP;

char *CONSTANT(omptarget_device_heap_buffer)
    __attribute__((used, retain, weak, visibility("protected")));

size_t CONSTANT(omptarget_device_heap_size)
    __attribute__((used, retain, weak, visibility("protected")));

namespace {
size_t HeapCurPos = 0;
mutex::TicketLock HeapLock;
}

extern "C" {

void *malloc(size_t Size) {
  mutex::LockGaurd LG(HeapLock);

  if (Size + HeapCurPos < omptarget_device_heap_size) {
    void *R = omptarget_device_heap_buffer + HeapCurPos;
    atomic::add(&HeapCurPos, Size, __ATOMIC_SEQ_CST);
    return R;
  }

  return nullptr;
}

void free(void *) {}
}

#pragma omp end declare target
