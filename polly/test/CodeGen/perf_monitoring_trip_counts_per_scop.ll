; RUN: opt %loadNPMPolly -passes=polly-codegen -polly-codegen-perf-monitoring \
; RUN:   -S < %s | FileCheck %s

; void f(long A[], long N) {
;   long i;
;   if (true)
;     for (i = 0; i < N; ++i)
;       A[i] = i;
; }
; void g(long A[], long N) {
;   long i;
;   if (true)
;     for (i = 0; i < N; ++i)
;       A[i] = i;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

define void @f(ptr %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %next

next:
  br i1 true, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %next], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}


define void @g(ptr %A, i64 %N) nounwind {
entry:
  fence seq_cst
  br label %next

next:
  br i1 true, label %for.i, label %return

for.i:
  %indvar = phi i64 [ 0, %next], [ %indvar.next, %for.i ]
  %scevgep = getelementptr i64, ptr %A, i64 %indvar
  store i64 %indvar, ptr %scevgep
  %indvar.next = add nsw i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %N
  br i1 %exitcond, label %return, label %for.i

return:
  fence seq_cst
  ret void
}

; Declaration of globals - Check for cycles declaration.
; CHECK: @"__polly_perf_in_f_from__%next__to__%polly.merge_new_and_old_trip_count" = weak thread_local(initialexec) constant i64 0
; CHECK: @"__polly_perf_in_g_from__%next__to__%polly.merge_new_and_old_trip_count" = weak thread_local(initialexec) constant i64 0

; Bumping up number of cycles in f
; CHECK:        %14 = load volatile i64, ptr @"__polly_perf_in_f_from__%next__to__%polly.merge_new_and_old_trip_count"
; CHECK-NEXT:   %15 = add i64 %14, 1
; CHECK-NEXT:   store volatile i64 %15, ptr @"__polly_perf_in_f_from__%next__to__%polly.merge_new_and_old_trip_count"

; Bumping up number of cycles in g
; CHECK:       %14 = load volatile i64, ptr @"__polly_perf_in_g_from__%next__to__%polly.merge_new_and_old_trip_count"
; CHECK-NEXT:  %15 = add i64 %14, 1
; CHECK-NEXT:  store volatile i64 %15, ptr @"__polly_perf_in_g_from__%next__to__%polly.merge_new_and_old_trip_count"
