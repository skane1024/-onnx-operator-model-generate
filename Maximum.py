import torch
import numpy as np


class Minimum(torch.nn.Module):
    def __init__(self, shapese):
        super(Minimum, self).__init__()
        self.constant = torch.nn.Parameter(torch.zeros((1,)))
    def forward(self, x):
        return torch.minimum(x,self.constant)
    
def gen_Minimum_op(input_shape, shapese):
    tinymodel = Minimum(shapese = (1,))
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"models/Minimum.onnx",export_params=True,input_names=["input"], output_names=["output"])


gen_Minimum_op((1,2,3,4,5),3)


