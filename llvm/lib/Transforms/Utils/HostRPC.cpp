//===- Transform/Utils/HostRPC.h - Code of automatic host rpc ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/HostRPC.h"

#include "llvm/ADT/EnumeratedArray.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"

using namespace llvm;

#define DEBUG_TYPE "host-rpc"

extern "C" int printf(const char *, ...);

__attribute__((weak)) Module *HostModule = nullptr;

namespace {
// TODO: Remove those functions implemented in device runtime.
static constexpr const char *InternalPrefix[] = {
    "__kmp", "llvm.", "nvm.", "omp_", "vprintf", "malloc", "free"};

std::string typeToString(Type *T) {
  if (T->is16bitFPTy())
    return "f16";
  if (T->isFloatTy())
    return "f32";
  if (T->isDoubleTy())
    return "f64";
  if (T->isPointerTy())
    return "ptr";
  if (T->isStructTy())
    return std::string(T->getStructName());
  if (T->isIntegerTy())
    return "i" + std::to_string(T->getIntegerBitWidth());

  llvm_unreachable("unknown type");
}

enum class HostRPCRuntimeFunction {
#define __OMPRTL_HOST_RPC(_ENUM) OMPRTL_##_ENUM
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_get_desc),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_add_arg),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_get_arg),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_send_and_wait),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_get_ret_val),
  __OMPRTL_HOST_RPC(__kmpc_host_rpc_set_ret_val),
  __OMPRTL_HOST_RPC(__last),
#undef __OMPRTL_HOST_RPC
};

#define __OMPRTL_HOST_RPC(_ENUM)                                               \
  auto OMPRTL_##_ENUM = HostRPCRuntimeFunction::OMPRTL_##_ENUM;
__OMPRTL_HOST_RPC(__kmpc_host_rpc_get_desc)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_add_arg)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_get_arg)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_send_and_wait)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_get_ret_val)
__OMPRTL_HOST_RPC(__kmpc_host_rpc_set_ret_val)
#undef __OMPRTL_HOST_RPC

enum OMPHostRPCArgType {
  // No need to copy.
  OMP_HOST_RPC_ARG_SCALAR = 0,
  OMP_HOST_RPC_ARG_PTR = 1,
  // Copy to host.
  OMP_HOST_RPC_ARG_PTR_COPY_TO = 2,
  // Copy to device
  OMP_HOST_RPC_ARG_PTR_COPY_FROM = 3,
  // TODO: Do we have a tofrom pointer?
  OMP_HOST_RPC_ARG_PTR_COPY_TOFROM = 4,
};

class HostRPC {
  LLVMContext &Context;
  // Device module
  Module &M;
  // Types
  Type *Int8PtrTy;
  Type *VoidTy;
  Type *Int32Ty;
  Type *Int64Ty;
  StructType *ArgInfoTy;

  struct CallSiteInfo {
    CallInst *CI = nullptr;
    SmallVector<Type *> Params;
  };

  struct HostRPCArgInfo {
    // OMPHostRPCArgType
    Constant *Type;
    Value *Size;
  };

  //
  SmallVector<Function *> HostEntryTable;

  EnumeratedArray<Function *, HostRPCRuntimeFunction,
                  HostRPCRuntimeFunction::OMPRTL___last>
      RFIs;

  static std::string getWrapperFunctionName(Function *F, CallSiteInfo &CSI) {
    std::string Name = "__kmpc_host_rpc_wrapper_" + std::string(F->getName());
    if (!F->isVarArg())
      return Name;

    for (unsigned I = F->getFunctionType()->getNumParams();
         I < CSI.Params.size(); ++I) {
      Name.push_back('_');
      Name.append(typeToString(CSI.Params[I]));
    }

    return Name;
  }

  static bool isInternalFunction(Function &F) {
    auto Name = F.getName();

    for (auto P : InternalPrefix)
      if (Name.startswith(P))
        return true;

    return false;
  }

  Value *convertToInt64Ty(IRBuilder<> &Builder, Value *V);

  Value *convertFromInt64TyTo(IRBuilder<> &Builder, Value *V, Type *TargetTy);

  // int device_wrapper(call_no, arg_info, ...) {
  //   void *desc = __kmpc_host_rpc_get_desc(call_no, num_args, arg_info);
  //   __kmpc_host_rpc_add_arg(desc, arg1, sizeof(arg1));
  //   __kmpc_host_rpc_add_arg(desc, arg2, sizeof(arg2));
  //   ...
  //   __kmpc_host_rpc_send_and_wait(desc);
  //   int r = (int)__kmpc_host_rpc_get_ret_val(desc);
  //   return r;
  // }
  Function *getDeviceWrapperFunction(StringRef WrapperName, Function *F,
                                     CallSiteInfo &CSI);

  // void host_wrapper(desc) {
  //   int arg1 = (int)__kmpc_host_rpc_get_arg(desc, 0);
  //   float arg2 = (float)__kmpc_host_rpc_get_arg(desc, 1);
  //   char *arg3 = (char *)__kmpc_host_rpc_get_arg(desc, 2);
  //   ...
  //   int r = actual_call(arg1, arg2, arg3, ...);
  //   __kmpc_host_rpc_set_ret_val(ptr(desc, (int64_t)r);
  // }
  Function *getHostWrapperFunction(StringRef WrapperName, Function *F,
                                   CallSiteInfo &CSI);

  bool rewriteWithHostRPC(Function *F);

public:
  HostRPC(Module &M) : Context(M.getContext()), M(M) {
    assert(&M.getContext() == &HostModule->getContext() &&
           "device and host modules have different context");

#define __OMP_TYPE(TYPE) TYPE = Type::get##TYPE(Context)
    __OMP_TYPE(Int8PtrTy);
    __OMP_TYPE(VoidTy);
    __OMP_TYPE(Int32Ty);
    __OMP_TYPE(Int64Ty);
#undef __OMP_TYPE

#define __OMP_RTL(_ENUM, MOD, VARARG, RETTY, ...)                              \
  {                                                                            \
    SmallVector<Type *> Params{__VA_ARGS__};                                   \
    FunctionType *FT = FunctionType::get(RETTY, Params, VARARG);               \
    Function *F = (MOD).getFunction(#_ENUM);                                   \
    if (!F)                                                                    \
      F = Function::Create(FT, GlobalValue::LinkageTypes::ExternalLinkage,     \
                           #_ENUM, (MOD));                                     \
    RFIs[OMPRTL_##_ENUM] = F;                                                  \
  }
    __OMP_RTL(__kmpc_host_rpc_get_desc, M, false, Int8PtrTy, Int32Ty, Int32Ty,
              Int8PtrTy)
    __OMP_RTL(__kmpc_host_rpc_add_arg, M, false, VoidTy, Int8PtrTy, Int64Ty,
              Int32Ty)
    __OMP_RTL(__kmpc_host_rpc_send_and_wait, M, false, VoidTy, Int8PtrTy)
    __OMP_RTL(__kmpc_host_rpc_get_ret_val, M, false, Int64Ty, Int8PtrTy)
    __OMP_RTL(__kmpc_host_rpc_get_arg, *HostModule, false, Int64Ty, Int8PtrTy,
              Int32Ty)
    __OMP_RTL(__kmpc_host_rpc_set_ret_val, *HostModule, false, VoidTy,
              Int8PtrTy, Int64Ty)
#undef __OMP_RTL

    ArgInfoTy = StructType::create({Int64Ty, Int64Ty}, "struct.arg_info_t");
  }

  bool run();
}; // namespace

Value *HostRPC::convertToInt64Ty(IRBuilder<> &Builder, Value *V) {
  Type *T = V->getType();

  if (T == Int64Ty)
    return V;

  if (T->isPointerTy())
    return Builder.CreatePtrToInt(V, Int64Ty);

  if (T->isIntegerTy())
    return Builder.CreateIntCast(V, Int64Ty, false);

  if (T->isFloatingPointTy()) {
    if (T->isFloatTy())
      V = Builder.CreateFPToSI(V, Int32Ty);
    return Builder.CreateFPToSI(V, Int64Ty);
  }

  llvm_unreachable("unknown cast to int64_t");
}

Value *HostRPC::convertFromInt64TyTo(IRBuilder<> &Builder, Value *V, Type *T) {
  if (T == Int64Ty)
    return V;

  if (T->isPointerTy())
    return Builder.CreateIntToPtr(V, T);

  if (T->isIntegerTy())
    return Builder.CreateIntCast(V, T, /* isSigned */ true);

  if (T->isFloatingPointTy()) {
    if (T->isFloatTy())
      V = Builder.CreateIntCast(V, Int32Ty, /* isSigned */ true);
    V = Builder.CreateSIToFP(V, T);
    return V;
  }

  llvm_unreachable("unknown cast from int64_t");
}

bool HostRPC::run() {
  bool Changed = false;

  SmallVector<Function *> WorkList;

  for (Function &F : M) {
    // If the function is already defined, it definitely does not require RPC.
    if (!F.isDeclaration())
      continue;

    // If it is an internal function, skip it as well.
    if (isInternalFunction(F))
      continue;

    // If there is no use of the function, skip it.
    if (F.use_empty())
      continue;

    WorkList.push_back(&F);
  }

  if (WorkList.empty())
    return Changed;

  for (Function *F : WorkList)
    Changed |= rewriteWithHostRPC(F);

  return Changed;
}

bool HostRPC::rewriteWithHostRPC(Function *F) {
  bool Changed = false;

  SmallVector<CallInst *> WorkList;

  for (User *U : F->users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;
    WorkList.push_back(CI);
  }

  if (WorkList.empty())
    return Changed;

  for (CallInst *CI : WorkList) {
    CallSiteInfo CSI;
    CSI.CI = CI;
    unsigned NumArgs = CI->arg_size();
    for (unsigned I = 0; I < NumArgs; ++I)
      CSI.Params.push_back(CI->getArgOperand(I)->getType());

    std::string WrapperName = getWrapperFunctionName(F, CSI);
    Function *DeviceWrapperFn = getDeviceWrapperFunction(WrapperName, F, CSI);
    Function *HostWrapperFn = getHostWrapperFunction(WrapperName, F, CSI);

    int32_t WrapperNumber = -1;
    for (unsigned I = 0; I < HostEntryTable.size(); ++I) {
      if (HostEntryTable[I] == HostWrapperFn) {
        WrapperNumber = I;
        break;
      }
    }
    if (WrapperNumber == -1) {
      WrapperNumber = HostEntryTable.size();
      HostEntryTable.push_back(HostWrapperFn);
    }

    DataLayout DL = M.getDataLayout();
    IRBuilder<> Builder(CI);

    auto CheckIfIdentifierPtr = [](const Value *V) {
      auto *CI = dyn_cast<CallInst>(V);
      if (!CI)
        return false;
      Function *Callee = CI->getCalledFunction();
      return Callee->getName().startswith("__kmpc_host_rpc_wrapper_");
    };

    auto CheckIfAlloca = [](const Value *V) {
      auto *CI = dyn_cast<CallInst>(V);
      if (!CI)
        return false;
      Function *Callee = CI->getCalledFunction();
      return Callee->getName() == "__kmpc_alloc_shared" ||
             Callee->getName() == "malloc";
    };

    SmallVector<HostRPCArgInfo> ArgInfos;
    bool IsConstantArgInfo = true;

    for (Value *Op : CI->args()) {
      if (!Op->getType()->isPointerTy()) {
        HostRPCArgInfo AI{
            ConstantInt::get(Int64Ty,
                             OMPHostRPCArgType::OMP_HOST_RPC_ARG_SCALAR),
            ConstantInt::getNullValue(Int64Ty)};
        ArgInfos.push_back(std::move(AI));
        continue;
      }

      Value *SizeVal = nullptr;
      OMPHostRPCArgType ArgType = OMP_HOST_RPC_ARG_PTR_COPY_TOFROM;

      SmallVector<const Value *> Objects;
      getUnderlyingObjects(Op, Objects);

      // TODO: Handle phi node
      if (Objects.size() != 1)
        llvm_unreachable("we can't handle phi node yet");

      auto *Obj = Objects.front();
      if (CheckIfIdentifierPtr(Obj)) {
        ArgType = OMP_HOST_RPC_ARG_SCALAR;
        SizeVal = ConstantInt::getNullValue(Int64Ty);
      } else if (CheckIfAlloca(Obj)) {
        auto *CI = dyn_cast<CallInst>(Obj);
        SizeVal = CI->getOperand(0);
        if (!isa<Constant>(SizeVal))
          IsConstantArgInfo = false;
      } else {
        if (auto *GV = dyn_cast<GlobalVariable>(Obj)) {
          SizeVal = ConstantInt::get(Int64Ty,
                                     DL.getTypeStoreSize(GV->getValueType()));
          if (GV->isConstant())
            ArgType = OMP_HOST_RPC_ARG_PTR_COPY_TO;
          if (GV->isConstant() && GV->hasInitializer()) {
            // TODO: If the global variable is contant, we can do some
            // optimization.
          }
        } else {
          // TODO: fix that when it occurs
          llvm_unreachable("cannot handle unknown type");
        }
      }

      HostRPCArgInfo AI{ConstantInt::get(Int64Ty, ArgType), SizeVal};
      ArgInfos.push_back(std::move(AI));
    }

    Value *ArgInfo = nullptr;

    if (!IsConstantArgInfo) {
      ArgInfo = Builder.CreateAlloca(
          ArgInfoTy, ConstantInt::get(Int64Ty, NumArgs), "arg_info");
      for (unsigned I = 0; I < NumArgs; ++I) {
        Value *AII = GetElementPtrInst::Create(
            ArrayType::get(ArgInfoTy, NumArgs), ArgInfo,
            {ConstantInt::getNullValue(Int64Ty), ConstantInt::get(Int64Ty, I)});
        Value *AIIType = GetElementPtrInst::Create(
            ArgInfoTy, AII, {ConstantInt::get(Int64Ty, 0)});
        Value *AIISize = GetElementPtrInst::Create(
            ArgInfoTy, AII, {ConstantInt::get(Int64Ty, 1)});
        Builder.Insert(AII);
        Builder.Insert(AIIType);
        Builder.Insert(AIISize);
        Builder.CreateStore(ArgInfos[I].Type, AIIType);
        Builder.CreateStore(ArgInfos[I].Size, AIISize);
      }
    } else {
      SmallVector<Constant *> ArgInfoInitVar;
      for (auto &AI : ArgInfos) {
        auto *CS =
            ConstantStruct::get(ArgInfoTy, {AI.Type, cast<Constant>(AI.Size)});
        ArgInfoInitVar.push_back(CS);
      }
      Constant *ArgInfoInit = ConstantArray::get(
          ArrayType::get(ArgInfoTy, NumArgs), ArgInfoInitVar);
      ArgInfo = new GlobalVariable(
          M, ArrayType::get(ArgInfoTy, NumArgs), /* isConstant */ true,
          GlobalValue::LinkageTypes::InternalLinkage, ArgInfoInit, "arg_info");
    }

    SmallVector<Value *> Args{ConstantInt::get(Int32Ty, WrapperNumber),
                              ArgInfo};
    for (Value *Op : CI->args())
      Args.push_back(Op);

    CallInst *NewCall = Builder.CreateCall(DeviceWrapperFn, Args);

    CI->replaceAllUsesWith(NewCall);
    CI->eraseFromParent();
  }

  F->eraseFromParent();

  return true;
}

Function *HostRPC::getDeviceWrapperFunction(StringRef WrapperName, Function *F,
                                            CallSiteInfo &CSI) {
  Function *WrapperFn = M.getFunction(WrapperName);
  if (WrapperFn)
    return WrapperFn;

  // return_type device_wrapper(int32_t call_no, void *arg_info, ...)
  SmallVector<Type *> Params{Int32Ty, Int8PtrTy};
  Params.append(CSI.Params);

  Type *RetTy = F->getReturnType();

  FunctionType *FT = FunctionType::get(RetTy, Params, /*isVarArg*/ false);
  WrapperFn = Function::Create(FT, GlobalValue::LinkageTypes::WeakODRLinkage,
                               WrapperName, M);

  // Emit the body of the device wrapper
  IRBuilder<> Builder(Context);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", WrapperFn);
  Builder.SetInsertPoint(EntryBB);

  // skip call_no and arg_info.
  constexpr const unsigned NumArgSkipped = 2;

  Value *Desc = nullptr;
  {
    Function *Fn = RFIs[OMPRTL___kmpc_host_rpc_get_desc];
    Desc = Builder.CreateCall(
        Fn,
        {WrapperFn->getArg(0),
         ConstantInt::get(Int32Ty, WrapperFn->arg_size() - NumArgSkipped),
         WrapperFn->getArg(1)},
        "desc");
  }

  {
    Function *Fn = RFIs[OMPRTL___kmpc_host_rpc_add_arg];
    for (unsigned I = NumArgSkipped; I < WrapperFn->arg_size(); ++I) {
      Value *V = convertToInt64Ty(Builder, WrapperFn->getArg(I));
      Builder.CreateCall(Fn, {Desc, V, ConstantInt::get(Int32Ty, I)});
    }
  }

  Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_send_and_wait], {Desc});

  if (RetTy->isVoidTy()) {
    Builder.CreateRetVoid();
    return WrapperFn;
  }

  Value *RetVal =
      Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_get_ret_val], {Desc});
  if (RetTy != RetVal->getType())
    RetVal = convertFromInt64TyTo(Builder, RetVal, RetTy);

  Builder.CreateRet(RetVal);

  return WrapperFn;
}

Function *HostRPC::getHostWrapperFunction(StringRef WrapperName, Function *F,
                                          CallSiteInfo &CSI) {
  Function *WrapperFn = HostModule->getFunction(WrapperName);
  if (WrapperFn)
    return WrapperFn;

  SmallVector<Type *> Params{Int8PtrTy};
  FunctionType *FT = FunctionType::get(VoidTy, Params, /* isVarArg */ false);
  WrapperFn = Function::Create(FT, GlobalValue::LinkageTypes::ExternalLinkage,
                               WrapperName, *HostModule);

  Value *Desc = WrapperFn->getArg(0);

  // Emit the body of the host wrapper
  IRBuilder<> Builder(Context);
  BasicBlock *EntryBB = BasicBlock::Create(Context, "entry", WrapperFn);
  Builder.SetInsertPoint(EntryBB);

  SmallVector<Value *> Args;
  for (unsigned I = 0; I < CSI.CI->arg_size(); ++I) {
    Value *V = Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_get_arg],
                                  {Desc, ConstantInt::get(Int32Ty, I)});
    Args.push_back(convertFromInt64TyTo(Builder, V, CSI.Params[I]));
  }

  // The host callee that will be called eventually by the host wrapper.
  Function *HostCallee = HostModule->getFunction(F->getName());
  if (!HostCallee)
    HostCallee = Function::Create(F->getFunctionType(), F->getLinkage(),
                                  F->getName(), *HostModule);

  Value *RetVal = Builder.CreateCall(HostCallee, Args);
  RetVal = convertToInt64Ty(Builder, RetVal);
  Builder.CreateCall(RFIs[OMPRTL___kmpc_host_rpc_set_ret_val], {Desc, RetVal});
  Builder.CreateRetVoid();

  return WrapperFn;
}
} // namespace

PreservedAnalyses HostRPCPass::run(Module &M, ModuleAnalysisManager &AM) {
  printf("[HostRPCPass] HostModule=%p\n", HostModule);
  if (!HostModule)
    return PreservedAnalyses::all();

  HostRPC RPC(M);
  bool Changed = RPC.run();

  return Changed ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
