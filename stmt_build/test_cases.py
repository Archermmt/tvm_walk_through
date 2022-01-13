import tvm

import sys
sys.path.append("..")
from visualize import PrimExprVisualizer

def cache_read(visualizer):
  print("\nTest cache_read")
  n = 1024
  dtype = "float32"
  A = tvm.te.placeholder((n, n), dtype=dtype, name='A')
  k = tvm.te.reduce_axis((0, n), name='k')
  B = tvm.te.compute((n,), lambda i: tvm.te.sum(A[i, k], axis=k), name='B')

  s = tvm.te.create_schedule(B.op)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/cache_read_before.prototxt")
  print("<Before>"+str(mod["main"]))

  #with cache_read
  AA = s.cache_read(A, "shared", [B])
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/cache_read_after.prototxt")
  print("<After>"+str(mod["main"]))

def compute_inline(visualizer):
  print("\nTest compute_inline")
  n = 1024
  k = 3
  pad = 2
  A = tvm.te.placeholder((n, n), name='A')
  W = tvm.te.placeholder((k, k), name='W')
  m = (n - k + 2 * pad) + 1
  Apad = tvm.te.compute((n + 2 * pad, n + 2 * pad),
                  lambda yy, xx: tvm.te.if_then_else(
                      tvm.te.all(yy >= pad, yy < pad + n, xx >= pad, xx < pad + n), 
                      A[yy - pad, xx - pad], tvm.tir.const(0., "float32")),
                      name='Apad')

  ry = tvm.te.reduce_axis((0, k), name='ry')
  rx = tvm.te.reduce_axis((0, k), name='rx')

  B = tvm.te.compute((m, m),
                  lambda yy, xx: 
                      tvm.te.sum(Apad[yy + ry, xx + rx] * W[ry, rx],
                      axis=[ry, rx]),
                      name='B')

  s = tvm.te.create_schedule(B.op)
  mod=tvm.driver.build_module.form_irmodule(s,[A,W,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/compute_inline_before.prototxt")
  print("<Before>"+str(mod["main"]))

  #with compute_inline
  s[Apad].compute_inline()
  mod=tvm.driver.build_module.form_irmodule(s,[A,W,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/compute_inline_after.prototxt")
  print("<After>"+str(mod["main"]))

def split(visualizer):
  print("\nTest split")
  n = 1024
  A = tvm.te.placeholder((n,), name='A')
  k = tvm.te.reduce_axis((0, n), name='k')
  B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')

  s = tvm.te.create_schedule(B.op)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/split_before.prototxt")
  print("<Before>"+str(mod["main"]))

  #with split
  ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/split_after.prototxt")
  print("<After>"+str(mod["main"]))

def tensorize(visualizer):
  print("\nTest tensorize")
  N, M, L = 1024, 512, 64
  A = tvm.te.placeholder((N, L), name='A')
  B = tvm.te.placeholder((M, L), name='B')
  k = tvm.te.reduce_axis((0, L), name='k')
  C = tvm.te.compute((N, M), lambda i, j: tvm.te.sum(A[i, k] * B[j, k], axis=k), name='C')
  s = tvm.te.create_schedule(C.op)

  def intrin_gemv(m, l):
    a = tvm.te.placeholder((l,), name='a')
    b = tvm.te.placeholder((m, l), name='b')
    k = tvm.te.reduce_axis((0, l), name='k')
    c =  tvm.te.compute((m,), lambda i: tvm.te.sum(a[k] * b[i, k], axis=k), name='c')
    Abuf = tvm.tir.decl_buffer(a.shape, a.dtype, name='A', offset_factor=1, strides=[1])
    Bbuf = tvm.tir.decl_buffer(b.shape, b.dtype, name='B', offset_factor=1, strides=[tvm.te.var("s1"), 1])
    Cbuf = tvm.tir.decl_buffer(c.shape, c.dtype, name='C', offset_factor=1, strides=[1])
    
    def intrin_func(ins, outs):
      ib = tvm.tir.ir_builder.create()
      aa, bb = ins
      cc = outs[0]
      ib.emit(tvm.tir.call_extern("int32", "gemv_update", cc.access_ptr("w"), aa.access_ptr("r"), bb.access_ptr("r"), m, l, bb.strides[0]))
      return ib.get()
    return tvm.te.decl_tensor_intrin(c.op, intrin_func, binds={a: Abuf, b: Bbuf, c: Cbuf})

  factor = 16
  x, y = C.op.axis
  z, = C.op.reduce_axis
  yo, yi = s[C].split(y, factor=factor)
  s[C].reorder(x, yo, yi, z)

  mod=tvm.driver.build_module.form_irmodule(s,[A,B,C],"main",binds=None)
  visualizer.visualize(mod,"visualizes/tensorize_before.prototxt")
  print("<Before>"+str(mod["main"]))

  #with tensorize
  gemv = intrin_gemv(factor, L)
  s[C].tensorize(yi, gemv)
  mod=tvm.driver.build_module.form_irmodule(s,[A,B,C],"main",binds=None)
  visualizer.visualize(mod,"visualizes/tensorize_after.prototxt")
  print("<After>"+str(mod["main"]))

def bind(visualizer):
  print("\nTest bind")
  n = 1024
  A = tvm.te.placeholder((n,), name='A')
  k = tvm.te.reduce_axis((0, n), name='k')
  B = tvm.te.compute((1,), lambda i: tvm.te.sum(A[k], axis=k), name='B')
  s = tvm.te.create_schedule(B.op)
  ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/bind_before.prototxt")
  print("<Before>"+str(mod["main"]))

  #with bind
  s[B].bind(ko, tvm.te.thread_axis("blockIdx.x"))
  s[B].bind(ki, tvm.te.thread_axis("threadIdx.x"))
  mod=tvm.driver.build_module.form_irmodule(s,[A,B],"main",binds=None)
  visualizer.visualize(mod,"visualizes/bind_after.prototxt")
  print("<After>"+str(mod["main"]))

if __name__=='__main__':
  visualizer=PrimExprVisualizer()
  cache_read(visualizer)
  compute_inline(visualizer)
  split(visualizer)
  bind(visualizer)
  tensorize(visualizer)