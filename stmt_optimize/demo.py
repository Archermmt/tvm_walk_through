import tvm

import sys
sys.path.append("..")
from visualize import PrimExprVisualizer

def simple(visualizer):
  print("\nTest StorageFlatten")
  n = tvm.te.var()
  A = tvm.te.placeholder((n, n), name='A')
  B = tvm.te.placeholder((n, n), name='B')
  C = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')
  s = tvm.te.create_schedule(C.op)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B,C],"main",binds=None)
  visualizer.visualize(mod,"visualizes/storage_flatten_before.prototxt")
  print("<Before>"+str(mod["main"]))
  
  #optimize
  pass_list=[tvm.tir.transform.StorageFlatten(64)]
  optimize = tvm.transform.Sequential(pass_list)
  mod = optimize(mod)
  visualizer.visualize(mod,"visualizes/storage_flatten_after.prototxt")
  print("<After>"+str(mod["main"]))

def vectorize(visualizer):
  print("\nTest VectorizeLoop")
  M = 1024
  N = 1024
  A = tvm.te.placeholder((M, N), name='A')
  B = tvm.te.placeholder((M, N), name='B')
  C = tvm.te.compute(
             (M, N),
             lambda x, y: A[x, y] + B[x, y],
             name='C')

  s = tvm.te.create_schedule(C.op)
  xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)
  s[C].vectorize(yi)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B,C],"main",binds=None)
  pass_list=[tvm.tir.transform.StorageFlatten(64)]
  optimize = tvm.transform.Sequential(pass_list)
  mod = optimize(mod)
  visualizer.visualize(mod,"visualizes/vectorize_loop_before.prototxt")
  print("<Before>"+str(mod["main"]))
  
  pass_list=[tvm.tir.transform.VectorizeLoop()]
  optimize = tvm.transform.Sequential(pass_list)
  mod = optimize(mod)
  visualizer.visualize(mod,"visualizes/vectorize_loop_after.prototxt")
  print("<After>"+str(mod["main"]))

def bind(visualizer):
  print("\nTest Simplify")
  n = 1024
  A = tvm.te.placeholder((n,), name='A')
  k = tvm.te.reduce_axis((0, n), name='k')
  B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')
  s = tvm.te.create_schedule(B.op)
  ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

  s[B].bind(ko, tvm.te.thread_axis("blockIdx.x"))
  s[B].bind(ki, tvm.te.thread_axis("threadIdx.x"))
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  pass_list=[tvm.tir.transform.StorageFlatten(64)]
  optimize = tvm.transform.Sequential(pass_list)
  mod = optimize(mod)
  visualizer.visualize(mod,"visualizes/simplify_before.prototxt")
  print("<Before>"+str(mod["main"]))
  
  pass_list=[tvm.tir.transform.Simplify()]
  optimize = tvm.transform.Sequential(pass_list)
  mod = optimize(mod)
  visualizer.visualize(mod,"visualizes/simplify_after.prototxt")
  print("<After>"+str(mod["main"]))

if __name__=='__main__':
  visualizer=PrimExprVisualizer(simple_mode=False)
  simple(visualizer)
  vectorize(visualizer)
  bind(visualizer)