import torch
import onnx
import numpy as np


class AddModel(torch.nn.Module):
    def forward(self, x,y):
        return torch.add(x,y)
    
class AddModelConstant(torch.nn.Module):
    def __init__(self, constant_shape):
        super(AddModelConstant, self).__init__()
        self.constant = torch.nn.Parameter(torch.zeros(constant_shape))
    def forward(self, x):
        return torch.add(x,self.constant)
    
    
def gen_add_op(input_shape1, input_shape2,constant=False):
    if constant:
        tinymodel = AddModelConstant(input_shape2)
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        torch.onnx.export(tinymodel,(a,),"Add_constant.onnx",export_params=True,input_names=["input"], output_names=["output"])
    else:
        tinymodel = AddModel()
        a=torch.FloatTensor(np.random.uniform(size = input_shape1))
        b=torch.FloatTensor(np.random.uniform(size = input_shape2))
        torch.onnx.export(tinymodel,(a,b),"Add.onnx",export_params=True, input_names=["input_1","input_2"], output_names=["output"],)



gen_add_op((1,2,3),(3,),False)
gen_add_op((1,2,3),(3,),True)


