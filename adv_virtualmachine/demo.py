import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine

import sys
sys.path.append("..")
from utils import array_des
from visualize import RelayVisualizer

def check_optimize(mod,target,params):
  visualizer=RelayVisualizer()
  with tvm.transform.PassContext(opt_level=3):
    compiler = relay.vm.VMCompiler()
    mod,params=compiler.optimize(mod, target=target, params=params)
  print("<Optimized>mod "+str(mod["main"]))
  visualizer.visualize(mod,path="visualizes/memory_opt.prototxt")

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = np.random.random_sample(shape_list[0][1]).astype('float32')
  graph = torch.jit.trace(model,torch.from_numpy(fake_input))

  #step 1 parse to relay
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  target = tvm.target.Target("llvm", host="llvm")
  
  #step 2.1.1 [optional] debug the optimize process
  #check_optimize(mod,target,params)
  
  #step 2 compile the module
  with tvm.transform.PassContext(opt_level=3):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

  #step 3 run the VirtualMachine
  dev = tvm.device("llvm", 0)
  vm = VirtualMachine(vm_exec, dev)
  vm.set_input("main", **{"input0": fake_input})
  res=vm.run()
  print("res "+array_des(res))