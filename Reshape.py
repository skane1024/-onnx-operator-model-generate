import torch
import numpy as np


class Reshape(torch.nn.Module):
    def __init__(self,shape):
        super(Reshape, self).__init__()
        self.shape = shape
    def forward(self, x):
        return torch.reshape(x,self.shape)
    
def gen_Reshape_op(input_shape, shape):
    tinymodel = Reshape(shape)
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"models/Reshape.onnx",export_params=True,input_names=["input"], output_names=["output"])
    

gen_Reshape_op([1,2,3,4],[2,12])