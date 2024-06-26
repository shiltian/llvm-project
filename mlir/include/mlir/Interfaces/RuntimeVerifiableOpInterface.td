//===- RuntimeVerifiableOpInterface.td - Op Verification ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_RUNTIMEVERIFIABLEOPINTERFACE
#define MLIR_INTERFACES_RUNTIMEVERIFIABLEOPINTERFACE

include "mlir/IR/OpBase.td"

def RuntimeVerifiableOpInterface : OpInterface<"RuntimeVerifiableOpInterface"> {
  let description = [{
    Implementations of this interface generate IR for runtime op verification.

    Incorrect op usage can often be caught by op verifiers based on static
    program information. However, in the absence of static program information,
    it can remain undetected at compile time (e.g., in case of dynamic memref
    strides instead of static memref strides). Such cases can be checked at
    runtime. The op-specific checks are generated by this interface.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Generate IR to verify this op at runtime, aborting runtime execution if
        verification fails.
      }],
      /*retTy=*/"void",
      /*methodName=*/"generateRuntimeVerification",
      /*args=*/(ins "::mlir::OpBuilder &":$builder,
                    "::mlir::Location":$loc)
    >,
  ];

  let extraClassDeclaration = [{
    /// Generate the error message that will be printed to the user when 
    /// verification fails.
    static std::string generateErrorMessage(Operation *op, const std::string &msg);
  }];
}

#endif // MLIR_INTERFACES_RUNTIMEVERIFIABLEOPINTERFACE
