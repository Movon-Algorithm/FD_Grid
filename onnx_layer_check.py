import onnx

def print_onnx_model_shapes(model_path):
    # ONNX Model file Load
    model = onnx.load(model_path)
    graph = model.graph

    # input layer shape
    print("Model inputs:")
    for input_tensor in graph.input:
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Name: {input_name}, Shape: {input_shape}")

    # output layer shape
    print("\nModel outputs:")
    for output_tensor in graph.output:
        output_name = output_tensor.name
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Name: {output_name}, Shape: {output_shape}")

# path for onnx file
model_path = 'mdfd.onnx'
print_onnx_model_shapes(model_path)
