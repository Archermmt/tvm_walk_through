import torch
import numpy as np
import torchvision.models as models

import tvm
from tvm import relay
from tvm import autotvm
from tvm.autotvm.tuner import GridSearchTuner

import sys
sys.path.append("..")
from visualize import RelayVisualizer

def check_optimize(mod,target,params):
  with tvm.transform.PassContext(opt_level=3):
    opt_mod, _ = relay.optimize(mod, target, params)
  visualizer=RelayVisualizer()
  visualizer.visualize(opt_mod,path="visualizes/optimized_mod.prototxt")
  print("optimized main func "+str(opt_mod["main"]))

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = np.random.random_sample(shape_list[0][1]).astype('float32')
  graph = torch.jit.trace(model,torch.from_numpy(fake_input))
  #step 1 parse
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  target = tvm.target.Target("llvm", host="llvm")
  #[optional] step 2.1 check optimize, for debug only
  #check_optimize(mod,target,params)
  #step 2 extract tasks
  tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
  #step 3 fintune
  runner = autotvm.LocalRunner(number=10,repeat=1,timeout=10,min_repeat_ms=0,)
  measure_option=autotvm.measure_option(
    builder=autotvm.LocalBuilder(n_parallel=1,build_func="default"), runner=runner)
  for i, task in enumerate(tasks[:1]):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = GridSearchTuner(task)
    tuner_obj.tune(
      n_trial=min(10, len(task.config_space)),
      early_stopping=100,
      measure_option=measure_option,
      callbacks=[
        autotvm.callback.progress_bar(10, prefix=prefix),
        autotvm.callback.log_to_file("autotvm_test.json"),
      ],
    )