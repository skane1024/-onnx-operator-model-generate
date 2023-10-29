import torch
import onnx
import numpy as np


class SubModel(torch.nn.Module):
    def forward(self, x,y):
        return torch.sub(x,y)
    
class SubModelConstant(torch.nn.Module):
    def __init__(self, constant_shape):
        super(SubModelConstant, self).__init__()
        self.constant = torch.nn.Parameter(torch.zeros(constant_shape))
    def forward(self, x):
        return torch.sub(x,self.constant)
    
    
def gen_Sub_op(input_shape1, input_shape2,constant=False):
    if constant:
        tinymodel = SubModelConstant(input_shape2)
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        torch.onnx.export(tinymodel,(a,),"Sub_constant.onnx",export_params=True,input_names=["input"], output_names=["output"])
    else:
        tinymodel = SubModel()
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        b=torch.FloatTensor(np.random.uniform(size = input_shape2))
        torch.onnx.export(tinymodel,(a,b),"Sub.onnx",export_params=True, input_names=["input_1","input_2"], output_names=["output"],)



gen_Sub_op((1,2,3),(3,),False)
gen_Sub_op((1,2,3),(3,),True)
