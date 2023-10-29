import torch
import torch.nn as nn

def gen_tanh_op(input_shape):
    model = nn.Tanh()
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"models/tanh" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
gen_tanh_op(input_shape = (1,3,224,224))
