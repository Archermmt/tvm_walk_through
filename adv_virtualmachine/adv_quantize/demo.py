import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm.contrib import graph_executor

import sys
sys.path.append("..")
from utils import array_des
from visualize import RelayVisualizer

def calibrate_dataset():
  for i in range(10):
    print("Creating {} th data".format(i))
    cal_data=np.random.random_sample((1,3,224,224)).astype('float32')
    yield {"input0": cal_data}

def check_optimize(mod,target,params):
  visualizer=RelayVisualizer()
  with tvm.transform.PassContext(opt_level=3):
    mod, params = relay.optimize(mod, params=params, target=target)
  print("<Optimized>mod "+str(mod["main"]))
  visualizer.visualize(mod,path="visualizes/quantized_opt.prototxt")

if __name__=='__main__':
  visualizer=RelayVisualizer()
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #step 1 parse
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  target = tvm.target.Target("llvm", host="llvm")
  #step 2 quantize
  with relay.quantize.qconfig(calibrate_mode="kl_divergence", weight_scale="max"):
    #step 2.1 [optional] debug the prerequisite_optimize process
    #mod=relay.quantize.prerequisite_optimize(mod,params)
    #visualizer.visualize(mod,path="visualizes/prerequisite.prototxt")
    mod = relay.quantize.quantize(mod, params, dataset=calibrate_dataset())
  visualizer.visualize(mod,path="visualizes/quantized.prototxt")
    
  #step 3.1 [optional] debug the optimize process
  #check_optimize(mod,target,params)

  #step 3 build lib
  with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)
  dev = tvm.cpu(0)
  m = graph_executor.GraphModule(lib["default"](dev))
  # Set inputs
  m.set_input("input0", tvm.nd.array(fake_input))
  # Execute
  m.run()
  # Get outputs
  res = m.get_output(0)
  print("output "+array_des(res))