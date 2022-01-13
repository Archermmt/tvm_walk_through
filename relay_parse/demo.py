import torch
import numpy as np
import torchvision.models as models

from tvm import relay

if __name__=='__main__':
  #prepare model and input
  model = models.resnet18(pretrained=True)
  shape_list = [("input0",(1,3,224,224))]
  fake_input = torch.from_numpy(np.random.random_sample(shape_list[0][1]).astype('float32'))
  graph = torch.jit.trace(model,fake_input)
  #main function
  mod, params = relay.frontend.from_pytorch(graph, shape_list)
  print("Parsed mod "+str(mod))