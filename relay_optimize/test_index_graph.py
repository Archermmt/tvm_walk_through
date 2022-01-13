import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay

def debug_optimize(mod,target,params):
  mod["main"]=relay.build_module.bind_params_by_name(mod["main"],params)
  #add transform passes
  seq = tvm.transform.Sequential(
    [
      relay.transform.SimplifyInference(),
    ]
  )
  mod=seq(mod)
  print("base func "+str(mod["main"]))
  
  seq = tvm.transform.Sequential(
    [
      relay.transform.FuseOps(),
    ]
  )
  mod=seq(mod)
  print("optimize func "+str(mod["main"]))
  
  return mod,params

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  #optimize the mod
  target = tvm.target.Target("llvm", host="llvm")
  with tvm.transform.PassContext(opt_level=3):
    mod,params=debug_optimize(mod,target,params)