import tensorflow as tf
try:
  tf_v1 = tf.compat.v1
except ImportError:
  tf_v1 = tf

import numpy as np

import tvm
from tvm import relay, runtime
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
import tvm.relay.testing.tf as tf_testing
from tvm.relay.op.contrib import tensorrt

import sys
sys.path.append("..")
from utils import array_des
from visualize import RelayVisualizer

def check_optimize(mod,params):
  with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
    mod, params = relay.optimize(mod, params=params, target="cuda")
  visualizer.visualize(mod,path="visualizes/optimized_mod.prototxt")

if __name__=='__main__':
  #prepare model and input
  model_url="https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/classify_image_graph_def-with_shapes.pb"
  model_name="classify_image_graph_def-with_shapes.pb"
  model_path = download_testdata(model_url, model_name, module=["tf", "InceptionV1"])
  visualizer = RelayVisualizer()
  #load graph and create input
  fake_input = np.random.random_sample([299,299,3]).astype('uint8')
  with tf_v1.gfile.GFile(model_path, "rb") as f:
    graph_def = tf_v1.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name="")
    graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    with tf_v1.Session() as sess:
      graph_def = tf_testing.AddShapesToGraphDef(sess, "softmax")

  #step 1 parse to relay
  shape_dict = {"DecodeJpeg/contents": fake_input.shape}
  mod, params = relay.frontend.from_tensorflow(graph_def, layout=None, shape=shape_dict)
  #visualizer.visualize(mod,path="visualizes/raw_mod.prototxt")

  #step 2 pre optimize for tensorrt
  mod, config = tensorrt.partition_for_tensorrt(mod,params)
  #visualizer.visualize(mod,path="visualizes/partition_graph.prototxt")

  #step 3.1 [optional] check optimize 
  #check_optimize(mod,params)

  #step 3 build the mod
  with tvm.transform.PassContext(opt_level=3, config={"relay.ext.tensorrt.options": config}):
    graph, lib, params = relay.build(mod, params=params, target="cuda")
    params = runtime.save_param_dict(params)

  #step 4 run executor and get results
  mod_ = graph_executor.create(graph, lib, device=tvm.gpu(0))
  mod_.load_params(params)
  mod_.run(data=fake_input)
  res = mod_.get_output(0)
  print("output "+array_des(res))
