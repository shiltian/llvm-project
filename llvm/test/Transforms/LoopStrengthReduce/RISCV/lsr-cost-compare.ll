; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=loop-reduce,loop-term-fold -S | FileCheck %s

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n64-S128"
target triple = "riscv64"

; a[] = 1.0
define void @test1(ptr %a) {
; CHECK-LABEL: @test1(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, ptr [[A:%.*]], i64 128000
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[LSR_IV1:%.*]] = phi ptr [ [[SCEVGEP:%.*]], [[LOOP]] ], [ [[A]], [[ENTRY:%.*]] ]
; CHECK-NEXT:    store float 1.000000e+00, ptr [[LSR_IV1]], align 4
; CHECK-NEXT:    [[SCEVGEP]] = getelementptr i8, ptr [[LSR_IV1]], i64 4
; CHECK-NEXT:    [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND:%.*]] = icmp eq ptr [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    br i1 [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %t15 = phi i64 [ 0, %entry ], [ %t20, %loop ]
  %t19 = getelementptr inbounds [32000 x float], ptr %a, i64 0, i64 %t15
  store float 1.0, ptr %t19, align 4
  %t20 = add nuw nsw i64 %t15, 1
  %t21 = icmp eq i64 %t20, 32000
  br i1 %t21, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; Same as test1, but with a use of a added outside the loop
define void @test2(ptr %a) {
; CHECK-LABEL: @test2(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SCEVGEP2:%.*]] = getelementptr i8, ptr [[A:%.*]], i64 128000
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[LSR_IV1:%.*]] = phi ptr [ [[SCEVGEP:%.*]], [[LOOP]] ], [ [[A]], [[ENTRY:%.*]] ]
; CHECK-NEXT:    store float 1.000000e+00, ptr [[LSR_IV1]], align 4
; CHECK-NEXT:    [[SCEVGEP]] = getelementptr i8, ptr [[LSR_IV1]], i64 4
; CHECK-NEXT:    [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND:%.*]] = icmp eq ptr [[SCEVGEP]], [[SCEVGEP2]]
; CHECK-NEXT:    br i1 [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    call void @use(ptr [[A]])
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %t15 = phi i64 [ 0, %entry ], [ %t20, %loop ]
  %t19 = getelementptr inbounds [32000 x float], ptr %a, i64 0, i64 %t15
  store float 1.0, ptr %t19, align 4
  %t20 = add nuw nsw i64 %t15, 1
  %t21 = icmp eq i64 %t20, 32000
  br i1 %t21, label %exit, label %loop

exit:                                             ; preds = %loop
  call void @use(ptr %a)
  ret void
}

; b[] = a[] + 1.0
define void @test3(ptr %a, ptr %b) {
; CHECK-LABEL: @test3(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr i8, ptr [[B:%.*]], i64 128000
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[LSR_IV2:%.*]] = phi ptr [ [[SCEVGEP3:%.*]], [[LOOP]] ], [ [[A:%.*]], [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[LSR_IV1:%.*]] = phi ptr [ [[SCEVGEP:%.*]], [[LOOP]] ], [ [[B]], [[ENTRY]] ]
; CHECK-NEXT:    [[T17:%.*]] = load float, ptr [[LSR_IV2]], align 4
; CHECK-NEXT:    [[T18:%.*]] = fadd float [[T17]], 1.000000e+00
; CHECK-NEXT:    store float [[T18]], ptr [[LSR_IV1]], align 4
; CHECK-NEXT:    [[SCEVGEP]] = getelementptr i8, ptr [[LSR_IV1]], i64 4
; CHECK-NEXT:    [[SCEVGEP3]] = getelementptr i8, ptr [[LSR_IV2]], i64 4
; CHECK-NEXT:    [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND:%.*]] = icmp eq ptr [[SCEVGEP]], [[SCEVGEP4]]
; CHECK-NEXT:    br i1 [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %t15 = phi i64 [ 0, %entry ], [ %t20, %loop ]
  %t16 = getelementptr inbounds [32000 x float], ptr %a, i64 0, i64 %t15
  %t17 = load float, ptr %t16, align 4
  %t18 = fadd float %t17, 1.000000e+00
  %t19 = getelementptr inbounds [32000 x float], ptr %b, i64 0, i64 %t15
  store float %t18, ptr %t19, align 4
  %t20 = add nuw nsw i64 %t15, 1
  %t21 = icmp eq i64 %t20, 32000
  br i1 %t21, label %exit, label %loop

exit:                                             ; preds = %loop
  ret void
}

; Same as test3, but with a use of both a and b outside the loop
define void @test4(ptr %a, ptr %b) {
; CHECK-LABEL: @test4(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SCEVGEP4:%.*]] = getelementptr i8, ptr [[B:%.*]], i64 128000
; CHECK-NEXT:    br label [[LOOP:%.*]]
; CHECK:       loop:
; CHECK-NEXT:    [[LSR_IV2:%.*]] = phi ptr [ [[SCEVGEP3:%.*]], [[LOOP]] ], [ [[A:%.*]], [[ENTRY:%.*]] ]
; CHECK-NEXT:    [[LSR_IV1:%.*]] = phi ptr [ [[SCEVGEP:%.*]], [[LOOP]] ], [ [[B]], [[ENTRY]] ]
; CHECK-NEXT:    [[T17:%.*]] = load float, ptr [[LSR_IV2]], align 4
; CHECK-NEXT:    [[T18:%.*]] = fadd float [[T17]], 1.000000e+00
; CHECK-NEXT:    store float [[T18]], ptr [[LSR_IV1]], align 4
; CHECK-NEXT:    [[SCEVGEP]] = getelementptr i8, ptr [[LSR_IV1]], i64 4
; CHECK-NEXT:    [[SCEVGEP3]] = getelementptr i8, ptr [[LSR_IV2]], i64 4
; CHECK-NEXT:    [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND:%.*]] = icmp eq ptr [[SCEVGEP]], [[SCEVGEP4]]
; CHECK-NEXT:    br i1 [[LSR_FOLD_TERM_COND_REPLACED_TERM_COND]], label [[EXIT:%.*]], label [[LOOP]]
; CHECK:       exit:
; CHECK-NEXT:    call void @use(ptr [[A]])
; CHECK-NEXT:    call void @use(ptr [[B]])
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %t15 = phi i64 [ 0, %entry ], [ %t20, %loop ]
  %t16 = getelementptr inbounds [32000 x float], ptr %a, i64 0, i64 %t15
  %t17 = load float, ptr %t16, align 4
  %t18 = fadd float %t17, 1.000000e+00
  %t19 = getelementptr inbounds [32000 x float], ptr %b, i64 0, i64 %t15
  store float %t18, ptr %t19, align 4
  %t20 = add nuw nsw i64 %t15, 1
  %t21 = icmp eq i64 %t20, 32000
  br i1 %t21, label %exit, label %loop

exit:                                             ; preds = %loop
  call void @use(ptr %a)
  call void @use(ptr %b)
  ret void
}

declare void @use(ptr)

