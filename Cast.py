import torch
import torch.nn as nn
import numpy as np
class TinyModel(torch.nn.Module):

    def forward(self,y):
        
        return y.type(torch.DoubleTensor)

tinymodel = TinyModel()

input = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))

output = input.type(torch.DoubleTensor)

print("This are the input tensors:"+str(input))
print("----------------------------------------------------")
print("This is the output:", output)
saveOnnx=True
loadModel=False
savePtModel = False


if savePtModel :
    torch.save({'model_state_dict':model.state_dict()}, name + ".pt")

if saveOnnx:
    torch.onnx.export(
            tinymodel,input,
            "Cast" + ".onnx",
            export_params=True
    )