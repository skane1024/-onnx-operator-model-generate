import torch
import torch.nn as nn

def gen_leakyrelu_op(input_shape,alpha):
    model = nn.LeakyReLU(alpha)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"LeakyRelu" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
    
gen_leakyrelu_op(input_shape = (1,3,224,224), alpha = 0.1)


