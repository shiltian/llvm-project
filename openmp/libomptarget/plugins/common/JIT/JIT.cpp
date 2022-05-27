//===-- JIT.cpp --- JIT module --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// JIT module for target plugins.
//
//===----------------------------------------------------------------------===//

#include "JIT.h"

#include "omptarget.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/LTO/legacy/LTOCodeGenerator.h"
#include "llvm/LTO/legacy/LTOModule.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <unordered_set>

#include "Debug.h"
#define TARGET_NAME JIT
#define DEBUG_PREFIX "JIT"

using namespace llvm;
using jit::impl::Action;

static codegen::RegisterCodeGenFlags RCGF;

namespace {

/// Init flag to call JIT::init().
std::once_flag InitFlag;

class TimeScope {
  const std::string ScopeName;
  std::chrono::system_clock::time_point Start;

public:
  TimeScope(const std::string &Scope)
      : ScopeName(Scope), Start(std::chrono::high_resolution_clock::now()) {}

  ~TimeScope() {
    auto Now = std::chrono::high_resolution_clock::now();
    auto Duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(Now - Start);
    std::cerr << ScopeName << ": " << Duration.count() << "ms\n";
  }
};

/// Cache for \p LLVMContext.
/// All resources allocated in LLVM are managed by \p LLVMContext. However, it
/// is not thread safe. To avoid data race and potential crash, we have to set
/// one context for each thread. This cache simply maps its thread id to a \p
/// LLVMContext object.
class LLVMContextCache {
  std::mutex Mutex;
  std::unordered_map<std::thread::id, std::unique_ptr<LLVMContext>> Map;

public:
  LLVMContext &get() {
    auto TId = std::this_thread::get_id();
    std::lock_guard<std::mutex> LG(Mutex);
    auto Itr = Map.find(TId);
    if (Itr != Map.end())
      return *Itr->second;
    auto C = std::make_unique<LLVMContext>();
    auto *P = C.get();
    Map[TId] = std::move(C);
    return *P;
  }
} ContextCache;

/// Cache for \p LTOModule.
/// It is a multi-level caching system. Since all resources allocated in LLVM
/// are managed by \p LLVMContext, modules allocated in one context can not be
/// passed to LLVM libraries using another one. As a result, the cache first
/// maps from the address of the \p LLVMContext to the second level cache, which
/// maps from the image start address to its corresponding \p LTOModule.
class LTOModuleCache {
  std::mutex Mutex;
  std::unordered_map<void *,
                     std::unordered_map<void *, std::unique_ptr<LTOModule>>>
      Map;

  LTOModule *getImpl(LLVMContext &Context, __tgt_device_image *Image) {
    auto I1 = Map.find(&Context);
    if (I1 == Map.end())
      return nullptr;

    auto I2 = I1->second.find(Image->ImageStart);
    if (I2 == I1->second.end())
      return nullptr;

    return I2->second.get();
  }

public:
  LTOModule *get(LLVMContext &Context, __tgt_device_image *Image) {
    std::lock_guard<std::mutex> LG(Mutex);
    return getImpl(Context, Image);
  }

  LTOModule *insert(LLVMContext &Context, __tgt_device_image *Image,
                    std::unique_ptr<LTOModule> M) {
    std::lock_guard<std::mutex> LG(Mutex);
    if (auto LM = getImpl(Context, Image))
      return LM;

    auto *P = M.get();
    auto &L = Map[&Context];
    L[Image->ImageStart] = std::move(M);

    return P;
  }
} ModuleCache;

/// Cache for \p TargetOptions.
class TargetOptionsCache {
  std::mutex Mutex;
  std::unordered_map<std::string, TargetOptions> Map;

public:
  TargetOptions &get(const std::string &T) {
    std::lock_guard<std::mutex> LG(Mutex);
    auto Itr = Map.find(T);
    if (Itr != Map.end())
      return Itr->second;
    Map[T] = codegen::InitTargetOptionsFromCodeGenFlags(Triple(T));
    return Map[T];
  }
} OptionsCache;

std::unique_ptr<LTOModule> createModuleFromImage(LLVMContext &Context,
                                                 __tgt_device_image *Image,
                                                 TargetOptions &Options) {
  auto *LM = ModuleCache.get(Context, Image);

  if (!LM) {
    DP("image " DPxMOD " not cached.\n", DPxPTR(Image));
    ptrdiff_t ImageSize = (char *)Image->ImageEnd - (char *)Image->ImageStart;
    auto ErrorOrModule = LTOModule::createFromBuffer(Context, Image->ImageStart,
                                                     ImageSize, Options);
    if (std::error_code Error = ErrorOrModule.getError()) {
      DP("failed to create LTOModule from buffer: %s.\n",
         Error.message().c_str());
      return nullptr;
    }

    LM = ModuleCache.insert(Context, Image, std::move(*ErrorOrModule));
  }

  auto ErrorOrModule = LTOModule::clone(*LM, Options);
  if (std::error_code Error = ErrorOrModule.getError()) {
    DP("failed to clone LTOModule: %s.\n", Error.message().c_str());
    return nullptr;
  }

  return std::move(*ErrorOrModule);
}

enum class ExecutionMode : uint8_t {
  Unknown,
  Generic,
  SPMD,
  SPMDGeneric,
};

ExecutionMode getExecutionMode(Module &M, StringRef KernelName) {
  GlobalVariable *ExecMode =
      M.getGlobalVariable(std::string(KernelName) + "_exec_mode");
  if (!ExecMode)
    return ExecutionMode::Unknown;
  assert(isa<ConstantInt>(ExecMode->getInitializer()) &&
         "ExecMode is not an integer!");
  int8_t ExecModeVal =
      cast<ConstantInt>(ExecMode->getInitializer())->getSExtValue();
  if (ExecModeVal == omp::OMP_TGT_EXEC_MODE_GENERIC_SPMD)
    return ExecutionMode::SPMDGeneric;
  if (ExecModeVal == omp::OMP_TGT_EXEC_MODE_SPMD)
    return ExecutionMode::SPMD;
  if (ExecModeVal == omp::OMP_TGT_EXEC_MODE_GENERIC)
    return ExecutionMode::Generic;

  return ExecutionMode::Unknown;
}

std::unordered_set<Action::ActionKind> DisabledOpts;

bool DumpModule = false;

void init() {
  // Initialize the configured targets.
  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  if (const char *Str = getenv("LIBOMPTARGET_JIT_DISABLED_OPTIMIZATIONS")) {
    auto ParseFn = [](const std::string &S) {
      if (S == "all") {
        DisabledOpts.insert(Action::ActionKind::Alignment);
        DisabledOpts.insert(Action::ActionKind::Specialization);
        DisabledOpts.insert(Action::ActionKind::NumTeams);
        DisabledOpts.insert(Action::ActionKind::NumThreads);
      } else if (S == "alignment") {
        DisabledOpts.insert(Action::ActionKind::Alignment);
      } else if (S == "specialization") {
        DisabledOpts.insert(Action::ActionKind::Specialization);
      } else if (S == "num_teams") {
        DisabledOpts.insert(Action::ActionKind::NumTeams);
      } else if (S == "num_threads") {
        DisabledOpts.insert(Action::ActionKind::NumThreads);
      }
    };

    std::string Tmp;
    auto P = Str;
    while (*P != '\0') {
      if (*P == ';') {
        ParseFn(Tmp);
        Tmp.clear();
      } else {
        Tmp.push_back(*P);
      }
      ++P;
    }
    if (!Tmp.empty())
      ParseFn(Tmp);
  }

  if (const char *Str = getenv("LIBOMPTARGET_JIT_DUMP_MODULE")) {
    if (!std::string(Str).empty())
      DumpModule = true;
  }
}

bool isOptEnabled(Action::ActionKind Kind) {
  if (DisabledOpts.find(Kind) != DisabledOpts.end()) {
    DP("optimization kind %d is disabled by env.\n", int(Kind));
    return false;
  }
  return true;
}

bool isOptEnabled(Action::ActionKind Kind,
                  const jit::impl::SpecializationStatistics &SS, int Index) {
  if (!isOptEnabled(Kind))
    return false;

  if (SS.reachThreshold(Kind, Index)) {
    DP("optimization kind %d on idx %d reaches threshold.\n", int(Kind), Index);
    return false;
  }

  return true;
}

bool isOptEnabled(Action::ActionKind Kind,
                  const jit::impl::SpecializationStatistics &SS) {
  if (!isOptEnabled(Kind))
    return false;

  if (SS.reachThreshold(Kind)) {
    DP("optimization kind %d reaches threshold.\n", int(Kind));
    return false;
  }

  return true;
}

bool compile(LLVMContext &Context, std::unique_ptr<LTOModule> LM,
             const std::string &MCpu, const TargetOptions &Options,
             CodeGenFileType FileType,
             const std::vector<std::string> &PreservedSymbols,
             std::string &OutputFileName) {
  LTOCodeGenerator CodeGen(Context);
  CodeGen.setDisableVerify(false);
  CodeGen.setCodePICModel(codegen::getExplicitRelocModel());
  CodeGen.setFreestanding(true);
  CodeGen.setDebugInfo(LTO_DEBUG_MODEL_NONE);
  CodeGen.setTargetOptions(Options);
  CodeGen.setShouldRestoreGlobalsLinkage(false);
  CodeGen.setCpu(MCpu);
  CodeGen.setAttrs(codegen::getMAttrs());
  CodeGen.setFileType(FileType);
  CodeGen.setUseDefaultPipeline(true);
  CodeGen.setOptLevel(3);

  CodeGen.addModule(LM.get());
  for (auto &S : PreservedSymbols)
    CodeGen.addMustPreserveSymbol(S);

  const char *Name;
  if (!CodeGen.compile_to_file(&Name)) {
    DP("failed to compile_fo_file.\n");
    return false;
  }

  OutputFileName = std::string(Name);

  if (DumpModule) {
    fprintf(stderr, ">>> after optimization\n");
    auto &M = CodeGen.getMergedModule();
    M.dump();
  }

  return true;
}

std::list<std::unique_ptr<MemoryBuffer>> PersistentBuffer;

void internalize(Module &M, LLVMContext &Context) {
  SmallVector<GlobalValue *, 4> Vec;
  auto *V = collectUsedGlobalVariables(M, Vec, /*CompilerUsed*/ false);
  SmallPtrSet<GlobalValue *, 4> Used{Vec.begin(), Vec.end()};

  if (auto *G = M.getGlobalVariable("IsSPMDMode")) {
    DP("internalize @IsSPMDMode and remove it from llvm.used.\n");
    G->setLinkage(GlobalValue::LinkageTypes::InternalLinkage);
    Used.erase(G);
  }

  if (Used.empty()) {
    V->eraseFromParent();
  } else {
    // Type of pointer to the array of pointers.
    PointerType *Int8PtrTy = Type::getInt8PtrTy(Context, 0);

    SmallVector<Constant *, 8> UsedArray;
    for (GlobalValue *GV : Used) {
      Constant *Cast =
          ConstantExpr::getPointerBitCastOrAddrSpaceCast(GV, Int8PtrTy);
      UsedArray.push_back(Cast);
    }
    // Sort to get deterministic order.
    array_pod_sort(UsedArray.begin(), UsedArray.end(),
                   [](Constant *const *A, Constant *const *B) {
                     Value *AStripped = (*A)->stripPointerCasts();
                     Value *BStripped = (*B)->stripPointerCasts();
                     return AStripped->getName().compare(BStripped->getName());
                   });
    ArrayType *ATy = ArrayType::get(Int8PtrTy, UsedArray.size());

    Module *M = V->getParent();
    V->removeFromParent();
    GlobalVariable *NV =
        new GlobalVariable(*M, ATy, false, GlobalValue::AppendingLinkage,
                           ConstantArray::get(ATy, UsedArray), "");
    NV->takeName(V);
    NV->setSection("llvm.metadata");
    delete V;
  }
}

enum AMDGPUMCpu : uint8_t {
  GK_GFX1036,
  GK_GFX1035,
  GK_GFX1034,
  GK_GFX1033,
  GK_GFX1032,
  GK_GFX1031,
  GK_GFX1030,
  GK_GFX1012,
  GK_GFX1011,
  GK_GFX1013,
  GK_GFX1010,
  GK_GFX940,
  GK_GFX90A,
  GK_GFX908,
  GK_GFX906,
  GK_GFX90C,
  GK_GFX909,
  GK_GFX904,
  GK_GFX902,
  GK_GFX900,
  GK_GFX810,
  GK_GFX805,
  GK_GFX803,
  GK_GFX802,
  GK_GFX801,
  GK_GFX705,
  GK_GFX704,
  GK_GFX703,
  GK_GFX702,
  GK_GFX701,
  GK_GFX700,
  GK_GFX602,
  GK_GFX601,
  GK_GFX600,
  GK_NONE,
};

AMDGPUMCpu parseAMDGPUMCpu(const std::string &MCpu) {
  static std::unordered_map<std::string, AMDGPUMCpu> Map = {
      {"gfx1036", GK_GFX1036}, {"gfx1035", GK_GFX1035}, {"gfx1034", GK_GFX1034},
      {"gfx1033", GK_GFX1033}, {"gfx1032", GK_GFX1032}, {"gfx1031", GK_GFX1031},
      {"gfx1030", GK_GFX1030}, {"gfx1012", GK_GFX1012}, {"gfx1011", GK_GFX1011},
      {"gfx1013", GK_GFX1013}, {"gfx1010", GK_GFX1010}, {"gfx940", GK_GFX940},
      {"gfx90A", GK_GFX90A},   {"gfx908", GK_GFX908},   {"gfx906", GK_GFX906},
      {"gfx90C", GK_GFX90C},   {"gfx909", GK_GFX909},   {"gfx904", GK_GFX904},
      {"gfx902", GK_GFX902},   {"gfx900", GK_GFX900},   {"gfx810", GK_GFX810},
      {"gfx805", GK_GFX805},   {"gfx803", GK_GFX803},   {"gfx802", GK_GFX802},
      {"gfx801", GK_GFX801},   {"gfx705", GK_GFX705},   {"gfx704", GK_GFX704},
      {"gfx703", GK_GFX703},   {"gfx702", GK_GFX702},   {"gfx701", GK_GFX701},
      {"gfx700", GK_GFX700},   {"gfx602", GK_GFX602},   {"gfx601", GK_GFX601},
      {"gfx600", GK_GFX600},
  };

  auto Itr = Map.find(MCpu);
  if (Itr == Map.end())
    return GK_NONE;

  return Itr->second;
}

void replaceTargetFeatures(Module &M, const std::string &MCpu) {
  if (M.getTargetTriple().find("amdgcn") == std::string::npos)
    return;

  for (Function &F : M.functions()) {
    bool IsSameMCpu = true;
    if (F.hasFnAttribute("target-cpu")) {
      auto A = F.getFnAttribute("target-cpu");
      if (A.getAsString() != MCpu) {
        IsSameMCpu = false;
        F.removeFnAttr("target-cpu");
        F.addFnAttr("target-cpu", MCpu);
      }
    }
    if (IsSameMCpu)
      return;
    if (F.hasFnAttribute("target-features")) {
      F.removeFnAttr("target-features");
      switch (parseAMDGPUMCpu(MCpu)) {
      case GK_GFX1036:
      case GK_GFX1035:
      case GK_GFX1034:
      case GK_GFX1033:
      case GK_GFX1032:
      case GK_GFX1031:
      case GK_GFX1030:
        F.addFnAttr("target-features", "+ci-insts");
        F.addFnAttr("target-features", "+dot1-insts");
        F.addFnAttr("target-features", "+dot2-insts");
        F.addFnAttr("target-features", "+dot5-insts");
        F.addFnAttr("target-features", "+dot6-insts");
        F.addFnAttr("target-features", "+dot7-insts");
        F.addFnAttr("target-features", "+dl-insts");
        F.addFnAttr("target-features", "+flat-address-space");
        F.addFnAttr("target-features", "+16-bit-insts");
        F.addFnAttr("target-features", "+dpp");
        F.addFnAttr("target-features", "+gfx8-insts");
        F.addFnAttr("target-features", "+gfx9-insts");
        F.addFnAttr("target-features", "+gfx10-insts");
        F.addFnAttr("target-features", "+gfx10-3-insts");
        F.addFnAttr("target-features", "+s-memrealtime");
        F.addFnAttr("target-features", "+s-memtime-inst");
        break;
      case GK_GFX1012:
      case GK_GFX1011:
        F.addFnAttr("target-features", "+dot1-insts");
        F.addFnAttr("target-features", "+dot2-insts");
        F.addFnAttr("target-features", "+dot5-insts");
        F.addFnAttr("target-features", "+dot6-insts");
        F.addFnAttr("target-features", "+dot7-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX1013:
      case GK_GFX1010:
        F.addFnAttr("target-features", "+dl-insts");
        F.addFnAttr("target-features", "+ci-insts");
        F.addFnAttr("target-features", "+flat-address-space");
        F.addFnAttr("target-features", "+16-bit-insts");
        F.addFnAttr("target-features", "+dpp");
        F.addFnAttr("target-features", "+gfx8-insts");
        F.addFnAttr("target-features", "+gfx9-insts");
        F.addFnAttr("target-features", "+gfx10-insts");
        F.addFnAttr("target-features", "+s-memrealtime");
        F.addFnAttr("target-features", "+s-memtime-inst");
        break;
      case GK_GFX940:
        F.addFnAttr("target-features", "+gfx940-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX90A:
        F.addFnAttr("target-features", "+gfx90a-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX908:
        F.addFnAttr("target-features", "+dot3-insts");
        F.addFnAttr("target-features", "+dot4-insts");
        F.addFnAttr("target-features", "+dot5-insts");
        F.addFnAttr("target-features", "+dot6-insts");
        F.addFnAttr("target-features", "+mai-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX906:
        F.addFnAttr("target-features", "+dl-insts");
        F.addFnAttr("target-features", "+dot1-insts");
        F.addFnAttr("target-features", "+dot2-insts");
        F.addFnAttr("target-features", "+dot7-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX90C:
      case GK_GFX909:
      case GK_GFX904:
      case GK_GFX902:
      case GK_GFX900:
        F.addFnAttr("target-features", "+gfx9-insts");
        LLVM_FALLTHROUGH;
      case GK_GFX810:
      case GK_GFX805:
      case GK_GFX803:
      case GK_GFX802:
      case GK_GFX801:
        F.addFnAttr("target-features", "+gfx8-insts");
        F.addFnAttr("target-features", "+16-bit-insts");
        F.addFnAttr("target-features", "+dpp");
        F.addFnAttr("target-features", "+s-memrealtime");
        LLVM_FALLTHROUGH;
      case GK_GFX705:
      case GK_GFX704:
      case GK_GFX703:
      case GK_GFX702:
      case GK_GFX701:
      case GK_GFX700:
        F.addFnAttr("target-features", "+ci-insts");
        F.addFnAttr("target-features", "+flat-address-space");
        LLVM_FALLTHROUGH;
      case GK_GFX602:
      case GK_GFX601:
      case GK_GFX600:
        F.addFnAttr("target-features", "+s-memtime-inst");
        break;
      case GK_NONE:
        break;
      default:
        llvm_unreachable("Unhandled GPU!");
      }
    }
  }
}
} // namespace

namespace jit {

namespace impl {

Action::Action(const std::string &S) {
  std::vector<std::string> Triple;
  std::string Tmp;
  for (auto C : S) {
    if (C == ':') {
      if (!Tmp.empty()) {
        Triple.push_back(Tmp);
        Tmp.clear();
      }
    } else {
      Tmp.push_back(C);
    }
  }
  if (!Tmp.empty())
    Triple.push_back(Tmp);

  // FIXME: If in the future we want to support nested actions, this will not
  // hold anymore.
  assert(Triple.size() == 3 && "broken action string");

  assert(Triple[POS_OpCode].size() == 1 && "opcode is not a char");

  char OpCode = Triple[POS_OpCode][0];
  const std::string &Idx = Triple[POS_Index];

  switch (OpCode) {
  case 's': {
    Kind = ActionKind::Specialization;
    Index = std::stoi(Idx);
    break;
  }
  case 'a': {
    Kind = ActionKind::Alignment;
    Index = std::stoi(Idx);
    break;
  }
  case 't': {
    Kind = ActionKind::NumThreads;
    assert(Idx == "n" && "index 'n' should be used for opcode 't'");
    break;
  }
  case 'T': {
    Kind = ActionKind::NumTeams;
    assert(Idx == "n" && "index 'n' should be used for opcode 'm'");
    break;
  }
  default:
    llvm_unreachable("unexpected operation code");
  }

  Value = std::stoll(Triple[POS_Value]);
}

Action::Action(ActionKind Kind, uintptr_t V, int Index)
    : Kind(Kind), Value(V), Index(Index) {
  switch (Kind) {
  case ActionKind::Specialization:
  case ActionKind::Alignment:
    break;
  default:
    llvm_unreachable("unexpected kind");
  }
}

Action::Action(ActionKind Kind, uintptr_t V) : Kind(Kind), Value(V) {
  switch (Kind) {
  case ActionKind::NumThreads:
  case ActionKind::NumTeams:
    break;
  default:
    llvm_unreachable("unexpected kind");
  }
}

std::string Action::toString() const {
  char OpCode;
  std::string Idx;

  switch (Kind) {
  case ActionKind::Specialization:
    OpCode = 's';
    Idx = std::to_string(Index);
    break;
  case ActionKind::Alignment:
    OpCode = 'a';
    Idx = std::to_string(Index);
    break;
  case ActionKind::NumThreads:
    OpCode = 't';
    Idx = "n";
    break;
  case ActionKind::NumTeams:
    OpCode = 'T';
    Idx = "n";
    break;
  default:
    llvm_unreachable("unknown kind");
  }

  std::string S;
  S.push_back(OpCode);
  S.push_back(':');
  S += Idx;
  S.push_back(':');
  S += std::to_string(Value);

  return S;
}

bool Action::match(const jit::Kernel &K) const {
  switch (Kind) {
  case ActionKind::Specialization: {
    if (Index >= K.getNumArgs())
      return false;

    return K.getArg(Index) == Value;
  }
  case ActionKind::Alignment: {
    if (Index >= K.getNumArgs())
      return false;

    return (K.getArg(Index) & Value) == 0;
  }
  case ActionKind::NumThreads:
    return Value == K.getNumThreads();
  case ActionKind::NumTeams:
    return Value == K.getNumTeams();
  default:
    llvm_unreachable("unknown kind");
  }

  return true;
}

std::string Action::ActionsToString(const std::vector<Action> &Actions) {
  std::string ActionString;
  for (unsigned I = 0; I < Actions.size(); ++I) {
    if (I != 0)
      ActionString.push_back('-');
    ActionString += Actions[I].toString();
  }
  return ActionString;
}

KernelSpecialization::KernelSpecialization(const std::string &Name,
                                           const std::string &MCpu,
                                           const std::string &ActionString)
    : KernelSpecialization(Name, MCpu) {
  std::string Tmp;
  for (auto Ch : ActionString) {
    if (Ch == '-') {
      Actions.emplace_back(Tmp);
      Tmp.clear();
    } else {
      Tmp.push_back(Ch);
    }
  }
  if (!Tmp.empty())
    Actions.emplace_back(Tmp);
}

KernelSpecialization::KernelSpecialization(const std::string &Name,
                                           const std::string &MCpu,
                                           const std::vector<Action> &A)
    : KernelSpecialization(Name, MCpu) {
  Actions = A;
}

bool KernelSpecialization::match(const Kernel &K) const {
  if (MCpu != K.getMCpu())
    return false;

  for (auto &A : Actions)
    if (!A.match(K))
      return false;

  return true;
}

bool SpecializationStatistics::reachThreshold(Action::ActionKind Kind,
                                              int Index) const {
  if (TotalCount < ThresholdTotalCount)
    return false;

  switch (Kind) {
  case Action::ActionKind::Alignment:
  case Action::ActionKind::Specialization:
    assert(Index < ArgCount.size() && "out of range access");
    return ArgCount[Index] / float(TotalCount) > ThresholdRatio;
    break;
  case Action::ActionKind::NumTeams:
  case Action::ActionKind::NumThreads:
    llvm_unreachable("unsupported action kind");
  default:
    llvm_unreachable("unknown action kind");
  }
}

bool SpecializationStatistics::reachThreshold(Action::ActionKind Kind) const {
  if (TotalCount < ThresholdTotalCount)
    return false;

  switch (Kind) {
  case Action::ActionKind::NumTeams:
    return NumTeamsCount / float(TotalCount) > ThresholdRatio;
  case Action::ActionKind::NumThreads:
    return NumThreadsCount / float(TotalCount) > ThresholdRatio;
  case Action::ActionKind::Alignment:
  case Action::ActionKind::Specialization:
    llvm_unreachable("unsupported action kind");
  default:
    llvm_unreachable("unknown action kind");
  }
}

bool TargetTable::match(const jit::Kernel &K) const {
  return Specialization->match(K);
}

__tgt_target_table *TargetTableCache::get(const jit::Kernel &K) const {
  auto Itr = Map.find(K.getName());
  if (Itr == Map.end())
    return nullptr;

  auto &L = Itr->second;
  for (auto &T : L)
    if (T.match(K))
      return T.get();

  return nullptr;
}

class StatisticsUpdater {
  SpecializationStatistics &SS;

public:
  StatisticsUpdater(SpecializationStatistics &SS) : SS(SS) {
    SS.TotalCount += 1;
  }

  void incForKind(Action::ActionKind Kind, int Index) {
    switch (Kind) {
    case Action::ActionKind::Alignment:
    case Action::ActionKind::Specialization:
      assert(Index < SS.ArgCount.size() && "out of range access");
      SS.ArgCount[Index]++;
      break;
    case Action::ActionKind::NumTeams:
    case Action::ActionKind::NumThreads:
      llvm_unreachable("unsupported action kind kind");
    default:
      llvm_unreachable("unknown action kind kind");
    }
  }

  void incForKind(Action::ActionKind Kind) {
    switch (Kind) {
    case Action::ActionKind::NumTeams:
      SS.NumTeamsCount++;
      break;
    case Action::ActionKind::NumThreads:
      SS.NumThreadsCount++;
      break;
    case Action::ActionKind::Alignment:
    case Action::ActionKind::Specialization:
      llvm_unreachable("unsupported action kind kind");
    default:
      llvm_unreachable("unknown action kind kind");
    }
  }
};

void Image::dump(std::ostream &OS) const {
  OS << Specialization.Name << '\0';
  OS << Specialization.MCpu << '\0';
  OS << Action::ActionsToString(Specialization.Actions) << '\0';
  uint32_t Size = End - Start;
  OS.write(reinterpret_cast<const char *>(&Size), sizeof(uint32_t));
  OS.write(Start, Size);
  OS << '\0';
}

ImageCache::ImageCache(const std::string &Arch) : Arch(Arch) {
  TimeScope TS(__PRETTY_FUNCTION__);

  std::string FileName = "libomptarget.jit." + Arch + ".cache";
  auto ErrorOrBuffer = MemoryBuffer::getFile(FileName);
  if (std::error_code Error = ErrorOrBuffer.getError())
    return;

  auto MB = std::move(*ErrorOrBuffer);
  const char *BufferStart = MB->getBufferStart();
  size_t Size = MB->getBufferSize();
  NewBuffer.push_back(std::move(MB));

  const char *P = BufferStart;
  // Verify architecture.
  {
    std::string A(P);
    P += A.size() + 1;
    if (A != Arch)
      return;
  }

  uint32_t NumKeys = *reinterpret_cast<const uint32_t *>(P);
  P += sizeof(uint32_t);

  for (unsigned I = 0; I < NumKeys; ++I) {
    std::string Key(P);
    P += Key.size() + 1;
    uint32_t NumImages = *reinterpret_cast<const uint32_t *>(P);
    P += sizeof(uint32_t);

    auto &L = Map[Key];
    for (unsigned J = 0; J < NumImages; ++J) {
      std::string KernelEntryName(P);
      P += KernelEntryName.size() + 1;
      std::string MCpu(P);
      P += MCpu.size() + 1;
      const std::string ActionString(P);
      P += ActionString.size() + 1;
      uint32_t KernelSize = *reinterpret_cast<const uint32_t *>(P);
      P += sizeof(uint32_t);
      const char *ImageStart = P;
      const char *ImageEnd = P + KernelSize;
      P = ImageEnd + 1;
      // Skip the tailing \0.
      ++P;
      KernelSpecialization KS(KernelEntryName, MCpu, ActionString);
      L.emplace_back(KS, ImageStart, ImageEnd);
    }
  }

  assert(P == BufferStart + Size + 1 && "corrupted offline buffer");
}

ImageCache::~ImageCache() {
  if (Map.empty())
    return;

  std::string FileName = "libomptarget.jit." + Arch + ".cache";
  std::ofstream OS(FileName);
  if (!OS)
    return;

  OS << Arch << '\0';

  {
    uint32_t Size = Map.size();
    OS.write(reinterpret_cast<const char *>(&Size), sizeof(uint32_t));
  }

  for (auto P : Map) {
    auto &Key = P.first;
    OS << Key << '\0';

    auto &L = P.second;
    uint32_t Size = L.size();
    OS.write(reinterpret_cast<const char *>(&Size), sizeof(uint32_t));

    for (auto &Img : L)
      Img.dump(OS);
  }
}

const Image *ImageCache::insert(const std::string &Key,
                                const KernelSpecialization &KS,
                                std::unique_ptr<llvm::MemoryBuffer> MB) {
  const char *ImageStart = MB->getBufferStart();
  const char *ImageEnd = MB->getBufferEnd();

  NewBuffer.push_back(std::move(MB));

  auto &Images = Map[Key];
  Images.emplace_back(KS, ImageStart, ImageEnd);

  return &Images.back();
}
} // namespace impl

JITEngine::JITEngine(const char *A, DeviceToolChain &DTC, int NumDevices)
    : Arch(A), NumDevices(NumDevices), DTC(DTC), DI(NumDevices),
      IC(new impl::ImageCache(A)), TTC(NumDevices) {}

__tgt_target_table *JITEngine::getTargetTable(int DeviceId, const Kernel &K) {
  if (DeviceId >= NumDevices) {
    DP("invalid device id: %d, # devices = %d.\n", DeviceId, NumDevices);
    return nullptr;
  }

  return TTC[DeviceId]->get(K);
}

__tgt_device_image *JITEngine::getImage(int DeviceId, Kernel &K,
                                        __tgt_device_image *Image) {
  TimeScope TS(__PRETTY_FUNCTION__);

  if (DeviceId >= NumDevices) {
    DP("invalid device id: %d, # devices = %d.\n", DeviceId, NumDevices);
    return nullptr;
  }

  DP("get image for kernel entry %s on device %d (arch = %s, mcpu = %s).\n",
     K.Name.c_str(), DeviceId, Arch.c_str(), DI[DeviceId].MCpu.c_str());

  std::string Key = DI[DeviceId].MCpu + "-" + K.getName();
  // FIXME: memory leak! Need to have an allocator in the future.
  __tgt_device_image *NewImage = new __tgt_device_image;
  *NewImage = *Image;

  if (auto *I = IC->get(Key, K)) {
    DP("found cached image with key = %s.\n", Key.c_str());
    auto P = I->get();
    NewImage->ImageStart = P.first;
    NewImage->ImageEnd = P.second;
    K.Id = (uintptr_t)&I->getKernelSpecialization();
    return NewImage;
  }

  DP("couldn't find cached image with key = %s.\n", Key.c_str());

  LLVMContext &Context = ContextCache.get();
  TargetOptions &Options = OptionsCache.get(Arch);

  auto LM = createModuleFromImage(Context, Image, Options);
  if (LM == nullptr)
    return nullptr;

  std::vector<Action> Actions;

  {
    auto &M = LM->getModule();
    auto F = M.getFunction(K.Name);
    if (!F) {
      DP("couldn't find kernel function %s from the module.\n", K.Name.c_str());
      return nullptr;
    }

    if (F->arg_size() != K.NumArgs) {
      DP("argument size mismatched: %d vs %d.\n", int(F->arg_size()),
         K.NumArgs);
      return nullptr;
    }

    impl::SpecializationStatistics &SS = Statistics.get(Key, K.NumArgs);
    impl::StatisticsUpdater SU(SS);

    if (DumpModule) {
      fprintf(stderr, ">>> before optimization for %s\n", K.Name.c_str());
      M.dump();
    }

    int NumThreads = 0;
    ExecutionMode ExecMode = getExecutionMode(M, K.getName());
    bool IsSPMDMode = ExecMode == ExecutionMode::SPMD ||
                      ExecMode == ExecutionMode::SPMDGeneric;
    bool IsSPMDGenericMode = ExecMode == ExecutionMode::SPMDGeneric;

    // TODO: This is just a WA.
    auto SpecializeIntrinsics = [&M, &Context](const std::string &Name,
                                               int Value) {
      auto *IF = M.getFunction(Name);
      if (!IF)
        return;
      for (auto *U : IF->users())
        if (auto *CI = dyn_cast<CallInst>(U)) {
          auto *C = ConstantInt::get(Type::getInt32Ty(Context), Value);
          CI->replaceAllUsesWith(C);
        }
    };

    if (isOptEnabled(Action::ActionKind::NumThreads, SS) &&
        !F->getFnAttribute("omp_target_thread_limit").isValid()) {
      if (IsSPMDMode) {
        NumThreads = K.NumThreads ? K.NumThreads : DI[DeviceId].NumThreads;
        if (NumThreads > DI[DeviceId].ThreadsPerBlock)
          NumThreads = DI[DeviceId].ThreadsPerBlock;
      }
      if (NumThreads) {
        DP("specialize num_threads = %d.\n", NumThreads);
        SU.incForKind(Action::ActionKind::NumThreads);

        F->addFnAttr("omp_target_thread_limit", std::to_string(NumThreads));
        // TODO:
        if (Arch.find("nvptx") != std::string::npos)
          SpecializeIntrinsics("llvm.nvvm.read.ptx.sreg.ntid.x", NumThreads);
        // We push K.NumThreads instead of NumThreads because for the same
        // kernel function, if K.NumThreads is same, NumThreads should be same
        // as well.
        Actions.emplace_back(Action::ActionKind::NumThreads, K.NumThreads);
      }
    }
    if (isOptEnabled(Action::ActionKind::NumTeams, SS) &&
        !F->getFnAttribute("omp_target_num_teams").isValid()) {
      int NumTeams = 0;
      if (K.NumTeams <= 0) {
        if (K.LoopTripCount > 0 && DI[DeviceId].EnvNumTeams < 0) {
          if (IsSPMDGenericMode) {
            assert(NumThreads && "NumThreads should not be zero at this point");
            NumTeams = K.LoopTripCount;
          } else if (IsSPMDMode) {
            assert(NumThreads && "NumThreads should not be zero at this point");
            NumTeams = ((K.LoopTripCount - 1) / NumThreads) + 1;
          }
        } else {
          NumTeams = DI[DeviceId].NumTeams;
        }
      } else {
        NumTeams = K.NumTeams;
      }

      if (NumTeams > DI[DeviceId].BlocksPerGrid)
        NumTeams = DI[DeviceId].BlocksPerGrid;

      if (NumTeams) {
        DP("specialize num_teams = %d.\n", NumTeams);
        SU.incForKind(Action::ActionKind::NumTeams);

        F->addFnAttr("omp_target_num_teams", std::to_string(NumTeams));
        // TODO:
        if (Arch.find("nvptx") != std::string::npos)
          SpecializeIntrinsics("llvm.nvvm.read.ptx.sreg.nctaid.x", NumTeams);
        // Same reason as K.NumThreads.
        Actions.emplace_back(Action::ActionKind::NumTeams, K.NumTeams);
        // TODO: We also set the range for function calls to
        // `llvm.nvvm.read.ptx.sreg.ctaid.x`.
        if (auto *Fn = M.getFunction("llvm.nvvm.read.ptx.sreg.ctaid.x")) {
          for (auto *U : Fn->users())
            if (auto *CI = dyn_cast<CallInst>(U)) {
              Metadata *RangeMD[] = {ConstantAsMetadata::get(ConstantInt::get(
                                         Type::getInt32Ty(Context), 0)),
                                     ConstantAsMetadata::get(ConstantInt::get(
                                         Type::getInt32Ty(Context), NumTeams))};
              CI->setMetadata(LLVMContext::MD_range,
                              MDNode::get(Context, RangeMD));
            }
        }
      }
    }

    int Idx = -1;

    for (auto &Arg : F->args()) {
      ++Idx;

      if (Arg.getType()->isPointerTy()) {
        if (!isOptEnabled(Action::ActionKind::Alignment, SS, Idx))
          continue;

        uintptr_t Ptr = (uintptr_t)K.Args[Idx];
        const static unsigned Alignments[] = {128, 64, 32, 16, 8};
        unsigned Alignment = 0;
        for (unsigned A : Alignments)
          if (!(Ptr & (A - 1))) {
            Alignment = A;
            break;
          }

        if (Alignment) {
          DP("set alignment of args[%d] to %u.\n", Idx, Alignment);
          SU.incForKind(Action::ActionKind::Alignment, Idx);

          Arg.addAttr(Attribute::get(Context, Attribute::Alignment, Alignment));
          Arg.addAttr(Attribute::get(Context, Attribute::NoUndef));
          Actions.emplace_back(impl::Action::ActionKind::Alignment, Alignment,
                               Idx);
        }

        continue;
      }

      if (isOptEnabled(Action::ActionKind::Specialization, SS, Idx)) {
        DP("set value of args[%d] to 0x%" PRIx64 ".\n", Idx, K.Args[Idx]);
        SU.incForKind(Action::ActionKind::Specialization, Idx);

        auto *C = ConstantInt::get(Type::getInt64Ty(Context), K.Args[Idx]);
        Arg.replaceAllUsesWith(C);
        Actions.emplace_back(Action::ActionKind::Specialization, K.Args[Idx],
                             Idx);
      }
    }
    // Final touch to the module before codegen.
    internalize(M, Context);
    replaceTargetFeatures(M, DI[DeviceId].MCpu);
  }

  jit::impl::KernelSpecialization KS(K.getName(), DI[DeviceId].MCpu, Actions);

  std::string OutputFileName;
  CodeGenFileType FileType = Arch.find("nvptx") != std::string::npos
                                 ? CodeGenFileType::CGFT_AssemblyFile
                                 : CodeGenFileType::CGFT_ObjectFile;
  if (!compile(Context, std::move(LM), DI[DeviceId].MCpu, Options, FileType,
               /* PreservedSymbols */ {K.getName(), K.getName() + "_exec_mode"},
               OutputFileName))
    return nullptr;

  auto DTCOutputBuffer = DTC.run(OutputFileName, DI[DeviceId]);
  if (!DTCOutputBuffer)
    return nullptr;

  auto *I = IC->insert(Key, KS, std::move(DTCOutputBuffer));
  assert(I && "failed to insert image to image cache");
  auto P = I->get();
  NewImage->ImageStart = P.first;
  NewImage->ImageEnd = P.second;
  K.Id = (uintptr_t)&I->getKernelSpecialization();

  return NewImage;
}

bool JITEngine::insertTargetTable(int DeviceId, const Kernel &K,
                                  __tgt_target_table *Table) {
  if (DeviceId >= NumDevices) {
    DP("invalid device id: %d, # devices = %d.\n", DeviceId, NumDevices);
    return false;
  }

  auto *KS = (const impl::KernelSpecialization *)K.Id;
  (void)TTC[DeviceId]->insert(KS, Table);

  return true;
}

__tgt_device_image *JITEngine::getImage(int DeviceId,
                                        __tgt_device_image *Image) {
  if (DeviceId >= NumDevices) {
    DP("invalid device id: %d, # devices = %d.\n", DeviceId, NumDevices);
    return nullptr;
  }

  DP("get full image on device %d (arch = %s, mcpu = %s).\n", DeviceId,
     Arch.c_str(), DI[DeviceId].MCpu.c_str());

  LLVMContext &Context = ContextCache.get();
  TargetOptions &Options = OptionsCache.get(Arch);

  auto LM = createModuleFromImage(Context, Image, Options);
  if (LM == nullptr)
    return nullptr;

  std::vector<std::string> Kernels;

  {
    Module &M = LM->getModule();

    if (DumpModule) {
      fprintf(stderr, ">>> before optimization\n");
      M.dump();
    }

    // Add all kernel entires to Kernels to make sure we preserve them.
    {
      NamedMDNode *MD = M.getOrInsertNamedMetadata("nvvm.annotations");
      if (!MD) {
        DP("failed to get named metadata nvvm.annotations.\n");
        return nullptr;
      }

      for (auto *Op : MD->operands()) {
        if (Op->getNumOperands() < 2)
          continue;
        MDString *KindID = dyn_cast<MDString>(Op->getOperand(1));
        if (!KindID || KindID->getString() != "kernel")
          continue;

        Function *KernelFn =
            mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
        if (!KernelFn)
          continue;

        Kernels.push_back(std::string(KernelFn->getName()));
        Kernels.push_back(std::string(KernelFn->getName()) + "_exec_mode");
      }
    }

    // Add all global variables that should be exposed to Kernels to make sure
    // we preserve them.
    {
      NamedMDNode *MD = M.getOrInsertNamedMetadata("omp_offload.info");
      if (MD) {
        for (auto *Op : MD->operands()) {
          if (Op->getNumOperands() < 2)
            continue;
          ConstantAsMetadata *KindID =
              dyn_cast<ConstantAsMetadata>(Op->getOperand(0));
          if (!KindID ||
              cast<ConstantInt>(KindID->getValue())->getZExtValue() != 1)
            continue;

          MDString *GV = dyn_cast<MDString>(Op->getOperand(1));
          if (!GV)
            continue;

          Kernels.push_back(std::string(GV->getString()));
        }
      }
    }

    internalize(M, Context);
    replaceTargetFeatures(M, DI[DeviceId].MCpu);
  }

  std::string OutputFileName;
  CodeGenFileType FileType = Arch.find("nvptx") != std::string::npos
                                 ? CodeGenFileType::CGFT_AssemblyFile
                                 : CodeGenFileType::CGFT_ObjectFile;
  if (!compile(Context, std::move(LM), DI[DeviceId].MCpu, Options, FileType,
               Kernels, OutputFileName))
    return nullptr;

  auto DTCBuffer = DTC.run(OutputFileName, DI[DeviceId]);
  if (!DTCBuffer)
    return nullptr;

  // FIXME: memory leak! Need to have an allocator in the future.
  __tgt_device_image *NewImage = new __tgt_device_image;
  *NewImage = *Image;
  NewImage->ImageStart = (void *)DTCBuffer->getBufferStart();
  NewImage->ImageEnd = (void *)DTCBuffer->getBufferEnd();

  PersistentBuffer.push_back(std::move(DTCBuffer));

  return NewImage;
}

void JITEngine::init() { std::call_once(InitFlag, ::init); }

Kernel Kernel::create(__tgt_device_image *Image, const char *Name,
                      const std::string &MCpu, void **Args, int NumArgs,
                      int NumTeams, int NumThreads, int LoopTripCount) {
  Kernel K;
  K.Name = std::string(Name);
  K.MCpu = MCpu;
  K.Args = (uintptr_t *)Args;
  K.NumArgs = NumArgs;

  if (isOptEnabled(Action::ActionKind::NumTeams)) {
    K.NumTeams = NumTeams;
    // The number of teams depends on loop trip count.
    K.LoopTripCount = LoopTripCount;
  }

  if (isOptEnabled(Action::ActionKind::NumThreads))
    K.NumThreads = NumThreads;

  return K;
}

bool JITEngine::isValidModule(const std::string &Arch,
                              __tgt_device_image *Image) {
  TimeScope TS(__PRETTY_FUNCTION__);

  TargetOptions &Options = OptionsCache.get(Arch);
  auto LM = createModuleFromImage(ContextCache.get(), Image, Options);
  // FIXME: This might not be enough but let's just keep it here for now.
  if (!LM)
    return false;
  return LM->getModule().getTargetTriple().find(Arch) != std::string::npos;
}

bool JITEngine::isSpecializationSupported(__tgt_device_image *Image) {
  for (auto *E = Image->EntriesBegin; E != Image->EntriesEnd; ++E)
    if (E->size)
      return false;

  if (const char *Str = getenv("LIBOMPTARGET_JIT_DISABLED_OPTIMIZATIONS"))
    if (std::string(Str) == "all")
      return false;

  return true;
}
} // namespace jit
