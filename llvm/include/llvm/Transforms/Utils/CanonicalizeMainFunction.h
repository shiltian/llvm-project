//===--------------- CanonicalizeMainFunction.h -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Utility function to canonicalize main function.
// The canonical main function is defined as: int main(int argc, char *argv[]);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_CANONICALIZEMAINFUNCTION_H
#define LLVM_TRANSFORMS_UTILS_CANONICALIZEMAINFUNCTION_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// A pass that canonicalizes main function in a module.
class CanonicalizeMainFunctionPass
    : public PassInfoMixin<CanonicalizeMainFunctionPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_CANONICALIZEMAINFUNCTION_H
