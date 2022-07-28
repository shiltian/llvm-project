//===--- Memory.h - OpenMP device runtime memory allocator -------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef OMPTARGET_MEMORY_H
#define OMPTARGET_MEMORY_H

#include "Types.h"

extern "C" {
__attribute__((leaf)) void *malloc(size_t Size);
__attribute__((leaf)) void free(void *Ptr);
}

#endif
