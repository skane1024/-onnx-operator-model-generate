import torch
import numpy as np


class Maximum(torch.nn.Module):
    def __init__(self, shapese):
        super(Maximum, self).__init__()
        self.constant = torch.nn.Parameter(torch.zeros((1,)))
    def forward(self, x):
        return torch.maximum(x,self.constant)
    
def gen_Maximum_op(input_shape, shapese):
    tinymodel = Maximum(shapese = (1,))
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"Maximum.onnx",export_params=True,input_names=["input"], output_names=["output"])


gen_Maximum_op((1,2,3,4,5),3)


