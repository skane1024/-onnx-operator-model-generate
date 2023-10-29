# import numpy as np
# import onnx


# def create_initializer_tensor(
#         name: str,
#         tensor_array: np.ndarray,
#         data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
# ) -> onnx.TensorProto:

#     # (TensorProto)
#     initializer_tensor = onnx.helper.make_tensor(
#         name=name,
#         data_type=data_type,
#         dims=tensor_array.shape,
#         vals=tensor_array.flatten().tolist())

#     return initializer_tensor


# def main() -> None:
#     # IO tensors (ValueInfoProto).
#     model_input_name = "X"
#     X = onnx.helper.make_tensor_value_info(model_input_name,
#                                            onnx.TensorProto.FLOAT,
#                                            [None, 3, 32, 32])
#     model_output_name = "Y"
    
#     Y = onnx.helper.make_tensor_value_info(model_output_name,
#                                            onnx.TensorProto.FLOAT,
#                                            [None, 1, 3])

import onnxruntime
import numpy
import os
import numpy as np
import matplotlib.pyplot as plt
import onnx
from onnx import helper
from onnx import TensorProto
from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

input1 = np.random.rand(1, 3, 4, 5).astype("float32")

# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 2])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.INT64, [2])

Shape_node = onnx.helper.make_node(
    name="Shape",  # Name is optional.
    op_type="Shape",
    inputs=['X'],
    outputs=['Y'],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [Shape_node],
    'test-model',
    [X],
    [Y],
)

 # Create the model (ModelProto)
model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
model_def.opset_import[0].version = 13

model_def = onnx.shape_inference.infer_shapes(model_def)

onnx.checker.check_model(model_def)

onnx.save(model_def, "Shape.onnx")

