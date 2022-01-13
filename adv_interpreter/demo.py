import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay

import sys
sys.path.append("..")
from utils import array_des
from visualize import RelayVisualizer

def check_optimize(inter):
  opt_mod=inter.optimize()
  visualizer=RelayVisualizer()
  visualizer.visualize(opt_mod,path="visualizes/optimized_mod.prototxt")

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = np.random.random_sample(shape_list[0][1]).astype('float32')
  graph = torch.jit.trace(model,torch.from_numpy(fake_input))
  #step 1 parse
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  target = tvm.target.Target("llvm", host="llvm")
  with tvm.transform.PassContext(opt_level=3):
    #step 2 optimize
    mod,params=relay.optimize(mod, target=target, params=params)
    #step 3 create Interpreter
    inter = relay.create_executor("debug", mod=mod, device=tvm.cpu(0), target=target)
    '''
    #[optional] step 3.1 optimize, only fo debug use
    check_optimize(inter)
    '''
    results = inter.evaluate()(fake_input)
  print("results "+array_des(results))