//===-- HeterogenousModuleUtils.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines some utility functions for heterognous module.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_HETEROGENOUSMODULEUTILS_H
#define LLVM_TRANSFORMS_UTILS_HETEROGENOUSMODULEUTILS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class Module;

namespace heterogenous {
/// Rename symbols in a module before getting merged into a heterogenous module.
void renameModuleSymbols(Module &SrcM, unsigned TargetId);
} // namespace heterogenous
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_HETEROGENOUSMODULEUTILS_H
