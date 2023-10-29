import torch
import torch.nn as nn

def gen_fc_op(input_shape,slope,bias=False):
    model = nn.Linear(input_shape, slope,bias)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"fc" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
    
gen_fc_op(input_shape = 100, slope = 67)




