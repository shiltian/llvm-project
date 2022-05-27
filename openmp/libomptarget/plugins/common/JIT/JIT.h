//===-- JIT.h --- JIT module ----------------------------------------------===//
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

#include <cassert>
#include <cstdint>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declaration.
struct __tgt_target_table;
struct __tgt_device_image;
struct __tgt_offload_entry;
struct __tgt_async_info;

namespace llvm {
class MemoryBuffer;
} // namespace llvm

namespace jit {
class Kernel;

namespace impl {

/// Optimization action applied to a kernel, which is in the form:
/// operation:index:value
///
/// 'operation' can be:
/// 's': value specialization;
/// 'a': alignment specialization;
/// 't': number of threads;
/// 'T': number of teams.
///
/// 'index' can be 'n' for those operations that don't require index, or an
/// integer number.
///
/// 'value' can be an action (recursively defined, but in fact we don't
/// support it for now), or an integer value.
class Action {
public:
  enum class ActionKind : uint8_t {
    None = 0,
    Alignment,
    Specialization,
    NumTeams,
    NumThreads,
  };

  explicit Action(const std::string &S);

  explicit Action(ActionKind AK, uintptr_t V, int Index);

  explicit Action(ActionKind AK, uintptr_t V);

  std::string toString() const;

  bool match(const Kernel &K) const;

  static std::string ActionsToString(const std::vector<Action> &Actions);

private:
  enum ValuePos : uint8_t {
    POS_OpCode = 0,
    POS_Index = 1,
    POS_Value = 2,
  };

  ActionKind Kind;
  uintptr_t Value;
  int Index;
};

class KernelSpecialization {
  /// Kernel entry name.
  const std::string Name;
  /// Target architecture.
  const std::string MCpu;
  ///
  std::vector<Action> Actions;

  friend class Image;

public:
  explicit KernelSpecialization(const std::string &Name,
                                const std::string &MCpu)
      : Name(Name), MCpu(MCpu) {}

  explicit KernelSpecialization(const std::string &Name,
                                const std::string &MCpu,
                                const std::string &ActionString);

  explicit KernelSpecialization(const std::string &Name,
                                const std::string &MCpu,
                                const std::vector<Action> &A);

  bool match(const Kernel &K) const;

  const std::string &getName() const { return Name; }
};

class SpecializationStatistics {
  /// Kernel name.
  const std::string KernelName;
  ///
  uint64_t ThresholdTotalCount = 20;
  ///
  float ThresholdRatio = 0.5f;
  /// Total number of specialization variants that have been generated for the
  /// corresponding kernel.
  uint64_t TotalCount = 0;
  /// Count for each argument.
  std::vector<uint64_t> ArgCount;
  /// Count for num_thread.
  uint64_t NumThreadsCount = 0;
  /// Count for num_team.
  uint64_t NumTeamsCount = 0;
  /// Gaurd lock.
  std::mutex Lock;

  friend class StatisticsUpdater;

public:
  SpecializationStatistics(const std::string &Name, int NumArgs)
      : KernelName(Name), ArgCount(NumArgs, 0) {}

  bool reachThreshold(Action::ActionKind Kind, int Index) const;

  bool reachThreshold(Action::ActionKind Kind) const;
};

class TargetTable {
  const KernelSpecialization *Specialization;
  __tgt_target_table *Table;

public:
  TargetTable(const KernelSpecialization *KS, __tgt_target_table *Table)
      : Specialization(KS), Table(Table) {}

  bool match(const Kernel &K) const;

  __tgt_target_table *get() const { return Table; }
};

class TargetTableCache {
  ///
  std::unordered_map<std::string, std::list<TargetTable>> Map;

public:
  __tgt_target_table *insert(const KernelSpecialization *KS,
                             __tgt_target_table *Table) {
    auto &Tables = Map[KS->getName()];
    Tables.emplace_back(KS, Table);

    return Tables.back().get();
  }

  __tgt_target_table *get(const Kernel &K) const;
};

class Image {
  KernelSpecialization Specialization;
  ///
  const char *Start = nullptr;
  ///
  const char *End = nullptr;

  void dump(std::ostream &OS) const;

  friend class ImageCache;

public:
  Image(const KernelSpecialization &KS, const char *ImageStart,
        const char *ImageEnd)
      : Specialization(KS), Start(ImageStart), End(ImageEnd) {}

  ///
  std::pair<void *, void *> get() const {
    return std::make_pair((void *)Start, (void *)End);
  }

  ///
  bool match(const Kernel &K) const { return Specialization.match(K); }

  const KernelSpecialization &getKernelSpecialization() const {
    return Specialization;
  }
};

class ImageCache {
public:
  ImageCache(const std::string &Arch);

  ~ImageCache();

  ///
  const Image *insert(const std::string &Key, const KernelSpecialization &KS,
                      std::unique_ptr<llvm::MemoryBuffer> MB);

  ///
  const Image *get(const std::string &Key, const Kernel &K) const {
    auto Itr = Map.find(Key);
    if (Itr == Map.end())
      return nullptr;

    auto &L = Itr->second;
    for (auto &I : L)
      if (I.match(K))
        return &I;

    return nullptr;
  }

private:
  const std::string Arch;
  ///
  std::list<std::unique_ptr<llvm::MemoryBuffer>> NewBuffer;
  ///
  std::unordered_map<std::string, std::list<Image>> Map;
};

} // namespace impl

struct DeviceInfo {
  /// Architecture, e.g. nvptx64, amdgcn.
  std::string Arch;
  /// GPU code name, e.g. sm_75 for Nvidia GPU.
  std::string MCpu;
  /// Maximum number of registers the device can support.
  uint64_t MaxNumRegs = 0;
  uint64_t ThreadsPerBlock = 0;
  uint64_t BlocksPerGrid = 0;
  uint64_t WarpSize = 32;
  /// Values set by users.
  int64_t EnvNumThreads = -1;
  int64_t EnvNumTeams = -1;
  /// Default values when users don't set explicitly.
  uint64_t NumThreads = 0;
  uint64_t NumTeams = 0;
};

class Kernel {
  /// Kernel entry name.
  std::string Name;
  /// Target architecture where the kernel is about to be launched.
  std::string MCpu;
  /// Number of threads.
  int NumThreads = 0;
  /// Number of teams.
  int NumTeams = 0;
  ///
  int LoopTripCount = 0;
  /// Number of arguments.
  int NumArgs = 0;
  /// Pointer to the kernel arguments.
  uintptr_t *Args = nullptr;
  /// If the kernel is specialized, an id will be assigned.
  uintptr_t Id = 0;

  Kernel() = default;

public:
  static Kernel create(__tgt_device_image *Image, const char *Name,
                       const std::string &MCpu, void **Args, int NumArgs,
                       int NumTeams, int NumThreads, int LoopTripCount);

  const std::string &getName() const { return Name; }

  const std::string &getMCpu() const { return MCpu; }

  int getNumThreads() const { return NumThreads; }

  int getNumTeams() const { return NumTeams; }

  uintptr_t getArg(int Index) const {
    assert(Index < NumArgs && "out of range access");
    return Args[Index];
  }

  int getNumArgs() const { return NumArgs; }

  friend class JITEngine;
};

class DeviceToolChain {
public:
  virtual std::unique_ptr<llvm::MemoryBuffer> run(const std::string &FileName,
                                                  const DeviceInfo &DI) = 0;
};

class JITEngine {
  const std::string Arch;
  int NumDevices = 0;

  DeviceToolChain &DTC;
  std::vector<DeviceInfo> DI;
  std::unique_ptr<impl::ImageCache> IC;
  std::vector<std::unique_ptr<impl::TargetTableCache>> TTC;

  class StatisticMap {
    std::unordered_map<std::string,
                       std::unique_ptr<impl::SpecializationStatistics>>
        Map;
    std::mutex Mtx;

  public:
    impl::SpecializationStatistics &get(const std::string &K, int NumArgs) {
      std::lock_guard<std::mutex> LG(Mtx);
      auto Itr = Map.find(K);
      if (Itr != Map.end())
        return *Itr->second;
      auto R = Map.insert(
          {K, std::make_unique<impl::SpecializationStatistics>(K, NumArgs)});
      return *R.first->second;
    }
  } Statistics;

public:
  JITEngine(const char *A, DeviceToolChain &DTC, int NumDevices);

  ///
  bool init(int DeviceId, const DeviceInfo &D) {
    if (DeviceId >= NumDevices)
      return false;
    DI[DeviceId] = D;
    TTC[DeviceId] = std::make_unique<impl::TargetTableCache>();
    return true;
  }

  /// Look up the target table cache. Return nullptr if there is no cache match
  /// for that specific kernel.
  __tgt_target_table *getTargetTable(int DeviceId, const Kernel &K);

  /// Get the device image.
  __tgt_device_image *getImage(int DeviceId, Kernel &K,
                               __tgt_device_image *Image);
  /// Get the device image without any kernel specialization.
  __tgt_device_image *getImage(int DeviceId, __tgt_device_image *Image);

  bool insertTargetTable(int DeviceId, const Kernel &K,
                         __tgt_target_table *Table);

  static bool isValidModule(const std::string &Arch, __tgt_device_image *Image);

  static bool isSpecializationSupported(__tgt_device_image *Image);

  static void init();
};
} // namespace jit
