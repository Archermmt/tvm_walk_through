import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm.relay.backend import graph_executor_codegen

import sys
sys.path.append("..")
from visualize import PrimExprVisualizer
visualizer=PrimExprVisualizer(simple_mode=False)
  
def process_codegen(target,optimize_mixed=False):
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  #optimize the mod
  with tvm.transform.PassContext(opt_level=3):
    mod, _ = relay.optimize(mod, target, params)
    grc = graph_executor_codegen.GraphExecutorCodegen(None, target)
    graph_json, lowered_func, params=grc.codegen(mod["main"])
  for tar, input_mod in lowered_func.items():
    if optimize_mixed:
      input_mod = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(input_mod)
      passes = [
        tvm.tir.transform.VerifyMemory(),
        tvm.tir.transform.ThreadSync("shared"),
        tvm.tir.transform.ThreadSync("warp"),
        tvm.tir.transform.InferFragment(),
        tvm.tir.transform.LowerThreadAllreduce(),
        tvm.tir.transform.MakePackedAPI(),
        tvm.tir.transform.SplitHostDevice(),
      ]
      input_mod = tvm.transform.Sequential(passes)(input_mod)
    return input_mod

def compare_mod(raw_mod,opt_mod,func_var=None,visual_name=None):
  #find a transformed var
  if not func_var:
    for var in raw_mod.get_global_vars():
      if var not in opt_mod.get_global_vars():
        continue
      try:
        raw_des=str(raw_mod[var])
        opt_des=str(opt_mod[var])
      except:
        continue
      if raw_des!=opt_des:
        func_var=var
        break
  if not func_var:
    print("raw mod and optimized mod are same")
    return
  print("<Before> {} : {}".format(func_var,raw_mod[func_var]))
  print("<After> {} : {}".format(func_var,opt_mod[func_var]))

  if visual_name:
    visualizer.visualize(raw_mod[func_var],"visualizes/{}_before.prototxt".format(visual_name))
    visualizer.visualize(opt_mod[func_var],"visualizes/{}_after.prototxt".format(visual_name))
  
def test_thread_sync(func_var=None):
  print("\nTest ThreadSync shared")
  target = tvm.target.Target("cuda")
  mod_mixed = process_codegen(target)
  raw_mixed = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_mixed)
  passes = [
    tvm.tir.transform.VerifyMemory(),
  ]
  raw_mixed = tvm.transform.Sequential(passes)(raw_mixed)
  opt_mixed = tvm.transform.Sequential([tvm.tir.transform.ThreadSync("shared")])(raw_mixed)
  compare_mod(raw_mixed,opt_mixed,func_var)

def test_make_packed_api(func_var=None):
  print("\nTest MakePackedAPI")
  target = tvm.target.Target("llvm")
  mod_mixed = process_codegen(target)
  raw_mixed = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_mixed)
  passes = [
    tvm.tir.transform.VerifyMemory(),
    tvm.tir.transform.ThreadSync("shared"),
    tvm.tir.transform.ThreadSync("warp"),
    tvm.tir.transform.InferFragment(),
    tvm.tir.transform.LowerThreadAllreduce(),
  ]
  raw_mixed = tvm.transform.Sequential(passes)(raw_mixed)
  opt_mixed = tvm.transform.Sequential([tvm.tir.transform.MakePackedAPI()])(raw_mixed)
  compare_mod(raw_mixed,opt_mixed,func_var,visual_name="make_packed_api")

def test_split_host_device(func_var=None):
  print("\nTest SplitHostDevice")
  target = tvm.target.Target("cuda")
  mod_mixed = process_codegen(target)
  raw_mixed = tvm.tir.transform.Apply(lambda f: f.with_attr("target", target))(mod_mixed)
  passes = [
    tvm.tir.transform.VerifyMemory(),
    tvm.tir.transform.ThreadSync("shared"),
    tvm.tir.transform.ThreadSync("warp"),
    tvm.tir.transform.InferFragment(),
    tvm.tir.transform.LowerThreadAllreduce(),
    tvm.tir.transform.MakePackedAPI()
  ]
  raw_mixed = tvm.transform.Sequential(passes)(raw_mixed)
  opt_mixed = tvm.transform.Sequential([tvm.tir.transform.SplitHostDevice()])(raw_mixed)
  print("<Before> fused_nn_dense_add : "+str(raw_mixed["fused_nn_dense_add"]))
  print("<After> fused_nn_dense_add "+str(opt_mixed["fused_nn_dense_add"]))
  print("<After> fused_nn_dense_add_kernel0 "+str(opt_mixed["fused_nn_dense_add_kernel0"]))
  visualizer.visualize(raw_mixed["fused_nn_dense_add"],"visualizes/split_host_device_before.prototxt")
  visualizer.visualize(opt_mixed["fused_nn_dense_add"],"visualizes/split_host_device_after_host.prototxt")
  visualizer.visualize(opt_mixed["fused_nn_dense_add_kernel0"],"visualizes/split_host_device_after_device.prototxt")

def test_lower_tvm_builtin(func_var=None):
  print("\nTest LowerTVMBuiltin")
  target = tvm.target.Target("cuda")
  target, target_host = tvm.target.Target.check_and_update_host_consist(target)
  mod_mixed = process_codegen(target,optimize_mixed=True)
  passes = [
    tvm.tir.transform.Filter(
        lambda f: "calling_conv" in f.attrs
        and f.attrs["calling_conv"].value != tvm.ir.CallingConv.DEVICE_KERNEL_LAUNCH
    ),
    tvm.tir.transform.Apply(lambda f: f.with_attr("target", target_host)),
  ]  
  raw_dev = tvm.transform.Sequential(passes)(mod_mixed)
  opt_dev = tvm.transform.Sequential([tvm.tir.transform.LowerTVMBuiltin(),])(raw_dev)
  compare_mod(raw_dev,opt_dev,func_var,visual_name="lower_tvm_builtin")

if __name__=='__main__':
  test_thread_sync()
  test_make_packed_api()
  test_split_host_device()
  test_lower_tvm_builtin("fused_nn_dense_add")