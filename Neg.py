import torch
import numpy as np


class Neg(torch.nn.Module):
    def __init__(self):
        super(Neg, self).__init__()
    def forward(self, x):
        return torch.neg(x)
    
def gen_Neg_op(input_shape):
    tinymodel = Neg()
    a=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(a,),"models/Neg.onnx",export_params=True,input_names=["input"], output_names=["output"])
    
    
    
gen_Neg_op([1,2,3,4])

