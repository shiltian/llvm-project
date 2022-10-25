//===------- HostRPC.cpp - Implementation of host RPC ------------- C++ ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Types.h"

#pragma omp begin declare target device_type(nohost)

extern "C" {
void *__kmpc_host_rpc_get_desc(int32_t CallNo, int32_t NumArgs, void *ArgInfo) {
  return nullptr;
}

void __kmpc_host_rpc_add_arg(void *Desc, int64_t Arg, int32_t ArgNum) {}

void __kmpc_host_rpc_send_and_wait(void *Desc) {}

int64_t __kmpc_host_rpc_get_ret_val(void *Desc) { return 0; }
}

#pragma omp end declare target
