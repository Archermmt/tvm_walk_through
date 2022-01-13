import tvm

import sys
sys.path.append("..")
from visualize import PrimExprVisualizer

if __name__=='__main__':
  visualizer=PrimExprVisualizer()
  n = tvm.te.var()
  A = tvm.te.placeholder((n, n), name='A')
  B = tvm.te.placeholder((n, n), name='B')
  #step 1 get ComputeOp
  C = tvm.te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')
  print("Compute struct "+str(C))
  visualizer.visualize(C,"visualizes/normal_compute.prototxt")

  #step 2 get Schedule
  s = tvm.te.create_schedule(C.op)

  #step 3 build stmt
  mod=tvm.driver.build_module.form_irmodule(s,[A,B,C],"main",binds=None)
  visualizer.visualize(mod,"visualizes/normal_stmt.prototxt")
  visualizer.visualize(mod,"visualizes/normal_stmt_complete.prototxt",simple_mode=False)
  print("Stmt struct "+str(mod["main"]))