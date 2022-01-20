import os
import torch
import numpy as np
import torchvision.models as models

import tvm
import tvm.testing
from tvm import relay
from tvm.contrib.msir.torch.transform import partition_for_torch
from tvm.contrib.msir.core.ir import MSIRGraphRuntimeModule
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
  model = models.resnet18(pretrained=True).eval()
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  golden = model(fake_input)
  graph = torch.jit.trace(model,fake_input)

  # partition main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list, with_name=True)
  mod, disabled_pass, config = partition_for_torch(mod,params)

  # optional optimize
  # debug_optimize(mod, params, disabled_pass, config)
  with tvm.transform.PassContext(opt_level=3, disabled_pass=disabled_pass, config=config):
    graph_json, mod1, params = relay.build(mod, target="llvm", params=params)

  msir_utils.set_work_dir("/tmp/dev_torch_test")
  module = MSIRGraphRuntimeModule(mod1)
  
  model = module.load_source()()
  state_dict = {k:torch.from_numpy(v) for k,v in module.load_weights().items()}
  model.load_state_dict(state_dict)
  res = model(fake_input)

  tvm.testing.assert_allclose(golden.detach().cpu().numpy(), res.detach().cpu().numpy(), rtol=5e-2)
  