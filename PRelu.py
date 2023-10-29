import torch
import torch.nn as nn

def gen_prelu_op(input_shape,slope):
    model = nn.PReLU(1, slope)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"PRelu" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
    
gen_prelu_op(input_shape = (1,3,224,224), slope = 0.1)


