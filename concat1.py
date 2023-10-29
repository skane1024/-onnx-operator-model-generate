import torch
import numpy as np


class Concat(torch.nn.Module):
    def __init__(self, axis):
        super(Concat, self).__init__()
        self.axis = axis
    def forward(self, x):
        return torch.concat(x,dim= self.axis)


def gen_concat_op(input_shapes, axis):
    tinymodel = Concat(axis=axis)
    input = []
    input_names = []
    for i,input_shape in enumerate(input_shapes):
        input.append(torch.FloatTensor(np.random.uniform(size = input_shape)))
        input_names.append(f"input_{str(i+1)}")
    torch.onnx.export(tinymodel,input,"concat.onnx",export_params=True,input_names = input_names, output_names=["output"])


gen_concat_op(((1,2,3),(1,2,3)),0)






