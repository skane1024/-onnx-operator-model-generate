import torch
import numpy as np


class Transpose(torch.nn.Module):
    def __init__(self,permute):
        super(Transpose, self).__init__()
        self.permute = permute
    def forward(self, x):
        return torch.permute(x,self.permute)
    
def gen_Transpose_op(input_shape, permute):
    tinymodel = Transpose(permute)
    input_data=torch.FloatTensor(np.random.uniform(size = input_shape))
    torch.onnx.export(tinymodel,(input_data,),"models/Transpose.onnx",export_params=True,input_names=["input"], output_names=["output"])
    
gen_Transpose_op([1,2,3,4],[0,3,1,2])
