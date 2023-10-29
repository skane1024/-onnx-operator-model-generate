import torch
import onnx
import numpy as np


class DivModel(torch.nn.Module):
    def forward(self, x,y):
        return torch.div(x,y)
    
class DivModelConstant(torch.nn.Module):
    def __init__(self, constant_shape):
        super(DivModelConstant, self).__init__()
        self.constant = torch.nn.Parameter(torch.zeros(constant_shape))
    def forward(self, x):
        return torch.div(x,self.constant)
    
    
def gen_Div_op(input_shape1, input_shape2,constant=False):
    if constant:
        tinymodel = DivModelConstant(input_shape2)
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        torch.onnx.export(tinymodel,(a,),"Div_constant.onnx",export_params=True,input_names=["input"], output_names=["output"])
    else:
        tinymodel = DivModel()
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        b=torch.FloatTensor(np.random.uniform(size = input_shape2))
        torch.onnx.export(tinymodel,(a,b),"Div.onnx",export_params=True, input_names=["input_1","input_2"], output_names=["output"],)



gen_Div_op((1,2,3),(3,),False)
gen_Div_op((1,2,3),(3,),True)
