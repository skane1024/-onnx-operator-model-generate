import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self, a, exp):
        
        return torch.pow(a, exp)

tinymodel = TinyModel()

x = np.array([[[1,2,3],[3,4,5]]])
y = np.array([[2,3,4],[2,3,4]])  #note the extra []

a=torch.FloatTensor(x)
exp=torch.FloatTensor(y)
output = torch.pow(a, exp)
print("This are the input tensors:"+str(a)+" & "+str(exp))
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
            (a,exp),
            "Pow_broadcast" + ".onnx",
            export_params=True
    )