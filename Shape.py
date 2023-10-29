# import torch
# import torch.nn as nn

# class TinyModel(torch.nn.Module):

#     def forward(self, x):
        
#         return torch._shape_as_tensor(x)

# tinymodel = TinyModel()

# input = torch.tensor([[1,0], [0,1]])
# output = torch._shape_as_tensor(input)
# # output = input.shape
# print("This is the input:",input)
# print("----------------------------------------------------")
# print("This is the output:", output)
# saveOnnx=True
# loadModel=False
# savePtModel = False

# # print("This is the output:", output)
# if savePtModel :
#     torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

# if saveOnnx:
#     torch.onnx.export(
#             tinymodel,
#             input,
#             "Shape" + ".onnx",
#             export_params=True
#     )

import numpy as np
import tensorflow as tf
from keras.preprocessing import image

import keras2onnx
import onnx

# load keras model
t = tf.Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])


model = tf.shape(t)

onnx_model = keras2onnx.convert_keras(model,model.name)

temp_model_file = 'shape1.onnx'
onnx.save_model(onnx_model, temp_model_file)
