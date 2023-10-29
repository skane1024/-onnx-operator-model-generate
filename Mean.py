import torch
import numpy as np


class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim = False):
        super(Mean, self).__init__()
        self.dim = dim
        self.keepdim=keepdim
    def forward(self, x):
        return torch.mean(x,self.dim, self.keepdim)
    
def gen_mean_op(input_shape, dim = 3):
    tinymodel = Mean(dim)
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"mean.onnx",export_params=True,input_names=["input"], output_names=["output"])


gen_mean_op((1,2,3,4,5),3)


