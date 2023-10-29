import torch
import torch.nn as nn




def gen_maxpool_op(input_shape, kernel_size,stride = None, padding = 0,dilation = 1,ceil_mode = False):
    model = nn.MaxPool2d(kernel_size=kernel_size, stride = stride,padding= padding,dilation = dilation,  ceil_mode = ceil_mode)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"MaxPool" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
    
    
gen_maxpool_op(input_shape = (1,3,224,224), kernel_size = (2,2),stride = None,padding = 1,dilation = 1,ceil_mode = False)