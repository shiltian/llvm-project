//===----- AbstractCallSiteTest.cpp - AbstractCallSite Unittests ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AbstractCallSiteTests", errs());
  return Mod;
}

TEST(AbstractCallSite, CallbackCall) {
  LLVMContext C;

  const char *IR =
      "define void @callback(i8* %X, i32* %A) {\n"
      "  ret void\n"
      "}\n"
      "define void @foo(i32* %A) {\n"
      "  call void (i32, void (i8*, ...)*, ...) @broker(i32 1, void (i8*, ...)* bitcast (void (i8*, i32*)* @callback to void (i8*, ...)*), i32* %A)\n"
      "  ret void\n"
      "}\n"
      "declare !callback !0 void @broker(i32, void (i8*, ...)*, ...)\n"
      "!0 = !{!1}\n"
      "!1 = !{i64 1, i64 -1, i1 true}";

  std::unique_ptr<Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  Function *Callback = M->getFunction("callback");
  ASSERT_NE(Callback, nullptr);

  const Use *CallbackUse = Callback->getSingleUndroppableUse();
  ASSERT_NE(CallbackUse, nullptr);

  AbstractCallSite ACS(CallbackUse);
  EXPECT_TRUE(ACS);
  EXPECT_TRUE(ACS.isCallbackCall());
  EXPECT_TRUE(ACS.isCallee(CallbackUse));
  EXPECT_EQ(ACS.getCalledFunction(), Callback);
}

TEST(AbstractCallSite, HeterogenousCallbackCall) {
  LLVMContext C;

  const char *IR =
      "@region_id = weak constant i8 0\n"
      "define void @callback1() {\n"
      "  ret void\n"
      "}\n"
      "define void @callback2() {\n"
      "  ret void\n"
      "}\n"
      "define void @foo(i32* %A) {\n"
      "  call void (i32, i8*, ...) @broker(i32 1, i8* @region_id, i32* %A)[\"region_id\"(void ()* @callback1, void ()* @callback2)]\n"
      "  ret void\n"
      "}\n"
      "declare !callback !0 void @broker(i32, i8*, ...)\n"
      "!0 = !{!1}\n"
      "!1 = !{i64 1, i64 -1, i1 true}";

  std::unique_ptr<Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  Function *CB1 = M->getFunction("callback1");
  Function *CB2 = M->getFunction("callback2");
  ASSERT_NE(CB1, nullptr);
  ASSERT_NE(CB2, nullptr);

  const Use *CB1Use = CB1->getSingleUndroppableUse();
  const Use *CB2Use = CB2->getSingleUndroppableUse();
  ASSERT_NE(CB1Use, nullptr);
  ASSERT_NE(CB2Use, nullptr);

  AbstractCallSite ACS1(CB1Use);
  AbstractCallSite ACS2(CB2Use);
  EXPECT_TRUE(ACS1);
  EXPECT_TRUE(ACS2);
  EXPECT_TRUE(ACS1.isCallbackCall());
  EXPECT_TRUE(ACS2.isCallbackCall());
  EXPECT_TRUE(ACS1.isCallee(CB1Use));
  EXPECT_TRUE(!ACS1.isCallee(CB2Use));
  EXPECT_TRUE(ACS2.isCallee(CB2Use));
  EXPECT_TRUE(!ACS2.isCallee(CB1Use));
  EXPECT_EQ(ACS1.getCalledFunction(), CB1);
  EXPECT_NE(ACS1.getCalledFunction(), CB2);
  EXPECT_EQ(ACS2.getCalledFunction(), CB2);
  EXPECT_NE(ACS2.getCalledFunction(), CB1);
}
