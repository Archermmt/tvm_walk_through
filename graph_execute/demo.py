import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import sys
sys.path.append("..")
from utils import array_des

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
    lib = relay.build(mod, target=target, params=params)
  #execute
  dev = tvm.cpu(0)
  m = graph_executor.GraphModule(lib["default"](dev))
  # Set inputs
  m.set_input("input0", tvm.nd.array(fake_input))
  # Execute
  m.run()
  # Get outputs
  res = m.get_output(0)
  print("output "+array_des(res))