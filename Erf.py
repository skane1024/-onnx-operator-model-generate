import torch
import numpy as np


class Erf(torch.nn.Module):
    def __init__(self):
        super(Erf, self).__init__()
    def forward(self, x):
        return torch.erf(x)
    
def gen_Erf_op(input_shape):
    tinymodel = Erf()
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"models/Erf.onnx",export_params=True,input_names=["input"], output_names=["output"])

