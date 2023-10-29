import torch
import torchvision

dummy_input = {-0.2717, -2.1627,  0.8192, -0.1963, -2.6961,  1.2576, -0.5996,  0.6201,
         1.0755, -0.6497,  0.9986,  0.5553,  1.5247,  0.6461,  0.0350, -0.7697,
         0.3247,  0.8656,  0.9730, -0.7275, -0.0262,  0.7698, -1.1662,  0.4401}
model = torchvision.models.alexnet(pretrained=True).cuda()


# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.
input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)