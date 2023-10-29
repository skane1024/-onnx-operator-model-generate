import torch
import numpy as np


class Split(torch.nn.Module):
    def __init__(self, axis, num_outputs):
        super(Split, self).__init__()
        self.axis = axis
        self.num_outputs = num_outputs
    def forward(self, x):
        return torch.split(x,self.num_outputs,self.axis)


def gen_Split_op(input_shapes, axis, num_outputs):
    tinymodel = Split(axis=axis,num_outputs=num_outputs)
    input=torch.FloatTensor(np.random.uniform(size = input_shapes))
    
    torch.onnx.export(tinymodel,input,"Split.onnx",export_params=True,input_names = ["input"], output_names=["output"])


gen_Split_op((1,2,3,4),3,2)






