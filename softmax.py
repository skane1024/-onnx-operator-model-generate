
import torch
import torch.nn as nn

def gen_softmax_op(input_shape,dim=1):
    model = nn.Softmax(dim=dim)
    input = torch.randn(input_shape)
    output = model(input)
    torch.onnx.export(model,input,"models/softmax" + ".onnx",export_params=True,input_names=["input"], output_names=["output"])
gen_softmax_op(input_shape = (1,3,224,224))