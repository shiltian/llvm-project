//===- HeterogenousModuleUtils.cpp - Utilities for heterogenous module -======//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements some utility functions for heterogenous module.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/HeterogenousModuleUtils.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace llvm {
namespace heterogenous {
void renameModuleSymbols(Module &SrcM, unsigned TargetId) {
  // Global variable
  for (GlobalVariable &GV : SrcM.globals()) {
    if (!isMangleNeeded(&GV))
      continue;
    GV.setName(mangleName(GV.getName(), TargetId));
    if (Comdat *C = GV.getComdat()) {
      Comdat *NC = SrcM.getOrInsertComdat(GV.getName());
      NC->setSelectionKind(C->getSelectionKind());
      GV.setComdat(NC);
    }
  }

  // Function
  for (Function &GV : SrcM.functions()) {
    if (!isMangleNeeded(&GV))
      continue;
    GV.setName(mangleName(GV.getName(), TargetId));
    if (Comdat *C = GV.getComdat()) {
      Comdat *NC = SrcM.getOrInsertComdat(GV.getName());
      NC->setSelectionKind(C->getSelectionKind());
      GV.setComdat(NC);
    }
  }

  // Alias
  for (GlobalAlias &GV : SrcM.aliases())
    GV.setName(mangleName(GV.getName(), TargetId));

  // IFunc
  for (GlobalIFunc &GV : SrcM.ifuncs())
    GV.setName(mangleName(GV.getName(), TargetId));
}
} // namespace heterogenous
} // namespace llvm
