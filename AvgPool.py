import torch
import torch.nn as nn

def gen_avgpool_op(input_shape, kernel_size,stride = None, padding = 0,ceil_mode = False):
    model = nn.AvgPool2d(kernel_size=kernel_size, stride = stride,padding= padding,  ceil_mode = ceil_mode)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"AvgPool2d" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
    
gen_avgpool_op(input_shape = (1,3,224,224), kernel_size = (2,2),stride = None,padding = 1,ceil_mode = False)