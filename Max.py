# import torch
# import torch.nn as nn

# class TinyModel(torch.nn.Module):

#     def forward(self, x, y):
        
#         return torch.maximum(x,y)

# tinymodel = TinyModel()

# a = torch.FloatTensor((1, 2, -1))
# b = torch.FloatTensor((3, 0, 4))
 
# output = torch.maximum(a, b)

# print("This are the input tensors:"+str(a)+"&"+str(b))
# print("----------------------------------------------------")
# print("This is the output:", output)
# saveOnnx=True
# loadModel=False
# savePtModel = False


# if savePtModel :
#     torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

# if saveOnnx:
#     torch.onnx.export(
#             tinymodel,(a,b),
#             "Max" + ".onnx",
#             export_params=True
#     )

import onnxruntime
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import TensorProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

# input1 = np.random.rand(1, 3, 4, 5).astype("float32")
# x = np.array([1, 2, -1]).astype("float32")
# y = np.array([3, 0, 4]).astype("float32") 
# Create one input (ValueInfoProto)
X1 = helper.make_tensor_value_info('X1', TensorProto.FLOAT,[1,3])
# Create one input (ValueInfoProto)
X2 = helper.make_tensor_value_info('X2', TensorProto.FLOAT,[1,3])
# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,3])

Shape_node = onnx.helper.make_node(
    name="Max",  # Name is optional.
    op_type="Max",
    inputs=['X1','X2'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [Shape_node],
    'test-model',
    [X1,X2],
    [Y],
)

 # Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
model_def.opset_import[0].version = 13

model_def = onnx.shape_inference.infer_shapes(model_def)

onnx.checker.check_model(model_def)

onnx.save(model_def, "Max.onnx")
