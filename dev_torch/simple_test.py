import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm.contrib.msir.torch.transform import partition_for_torch
from tvm.contrib.msir.core import utils as msir_utils

def debug_optimize(mod, params, disabled_pass, config):
  with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_pass, config=config):
    mod, params=relay.optimize(mod, params=params, target="llvm")
    mod = relay.transform.InferLayout()(mod)
  from tvm.contrib.msir.core.ir import build_from_relay
  graph=build_from_relay(mod,"main")
  graph.visualize("visualizes/test.prototxt")
  
if __name__=='__main__':
  # prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  # partition main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list, with_name=True)
  mod, disabled_pass, config = partition_for_torch(mod,params)

  # optional optimize
  # debug_optimize(mod, params, disabled_pass, config)
  with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_pass, config=config):
    graph_json, mod1, params = relay.build(mod, target='llvm', params=params)

  '''
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
  '''