//===-------------- CanonicalizeMainFunction.cpp ----------------*- C++ -*-===//
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

#include "llvm/Transforms/Utils/CanonicalizeMainFunction.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "canonicalize-main-function"

static cl::opt<std::string>
    MainFunctionName("canonical-main-function-name",
                     cl::desc("New main function name"),
                     cl::value_desc("main function name"));

bool rewriteMainFunction(Function &F) {
  if (F.arg_size() == 2 && F.getReturnType()->isIntegerTy(32))
    return false;

  auto &Ctx = F.getContext();
  auto &DL = F.getParent()->getDataLayout();
  auto *Int32Ty = IntegerType::getInt32Ty(Ctx);
  auto *PtrTy = PointerType::get(Ctx, DL.getDefaultGlobalsAddressSpace());

  FunctionType *NewFnTy =
      FunctionType::get(Int32Ty, {Int32Ty, PtrTy}, /* isVarArg */ false);
  Function *NewFn =
      Function::Create(NewFnTy, F.getLinkage(), F.getAddressSpace(), "");
  F.getParent()->getFunctionList().insert(F.getIterator(), NewFn);
  NewFn->takeName(&F);
  NewFn->copyAttributesFrom(&F);
  NewFn->setSubprogram(F.getSubprogram());
  F.setSubprogram(nullptr);
  NewFn->splice(NewFn->begin(), &F);

  if (!F.getReturnType()->isIntegerTy(32)) {
    SmallVector<ReturnInst *> WorkList;
    for (BasicBlock &BB : *NewFn)
      for (Instruction &I : BB) {
        auto *RI = dyn_cast<ReturnInst>(&I);
        if (!RI)
          continue;
        assert(RI->getReturnValue() == nullptr &&
               "return value of a void main function is not nullptr");
        WorkList.push_back(RI);
      }
    for (auto *RI : WorkList) {
      (void)ReturnInst::Create(Ctx, ConstantInt::getNullValue(Int32Ty), RI);
      RI->eraseFromParent();
    }
  }

  if (F.arg_size() == NewFn->arg_size())
    for (unsigned I = 0; I < NewFn->arg_size(); ++I) {
      Argument *OldArg = F.getArg(I);
      Argument *NewArg = NewFn->getArg(I);
      NewArg->takeName(OldArg);
      OldArg->replaceAllUsesWith(NewArg);
    }

  return true;
}

PreservedAnalyses CanonicalizeMainFunctionPass::run(Module &M,
                                                    ModuleAnalysisManager &AM) {
  Function *MainFunc = nullptr;

  for (Function &F : M)
    if (F.getName() == "main") {
      assert(MainFunc == nullptr && "more than one main function");
      MainFunc = &F;
    }

  if (MainFunc == nullptr)
    return PreservedAnalyses::all();

  bool Changed = false;

  if (!MainFunctionName.empty() && MainFunc->getName() != MainFunctionName) {
    MainFunc->setName(MainFunctionName);
    Changed = true;
  }

  if (rewriteMainFunction(*MainFunc)) {
    MainFunc->eraseFromParent();
    Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
