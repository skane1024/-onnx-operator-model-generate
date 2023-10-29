import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, a, b):
        
        return torch.eq(a, b)

tinymodel = TinyModel()

x = np.array([1,2,3])
y = np.array([4,2,6])  

a=torch.FloatTensor(x)
b=torch.FloatTensor(y)
output = torch.eq(a, b)
print("This are the input tensors:"+str(a)+" & "+str(b))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,
            (a,b),
            "Equal" + ".onnx",
            export_params=True
    )