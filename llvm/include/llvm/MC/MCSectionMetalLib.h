//===- MCSectionMetalLib.h - MetalLib Machine Code Sections -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionMetalLib class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTIONMetalLib_H
#define LLVM_MC_MCSECTIONMetalLib_H

#include "llvm/MC/MCSection.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {

class MCSymbol;

class MCSectionMetalLib final : public MCSection {
  friend class MCContext;

  MCSectionMetalLib(SectionKind K, MCSymbol *Begin)
      : MCSection(SV_MetalLib, "", K, Begin) {}
  // TODO: Add StringRef Name to MCSectionMetalLib.

public:
  ~MCSectionMetalLib() = default;
  void printSwitchToSection(const MCAsmInfo &MAI, const Triple &T,
                            raw_ostream &OS,
                            const MCExpr *Subsection) const override {}
  bool useCodeAlign() const override { return false; }
  bool isVirtualSection() const override { return false; }
};

} // end namespace llvm

#endif // LLVM_MC_MCSECTIONMetalLib_H
