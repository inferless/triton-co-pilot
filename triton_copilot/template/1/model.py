import os
import sys
import ##file## as app
import json
import triton_python_backend_utils as pb_utils
import numpy as np
import uuid

custom_model = app.##model_class##()

model_data_type_maping = {
    bool: "TYPE_BOOL",
    np.uint8: "TYPE_UINT8",
    np.uint16: "TYPE_UINT16",
    np.uint32: "TYPE_UINT32",
    np.uint64: "TYPE_UINT64",
    np.int8: "TYPE_INT8",
    np.int16: "TYPE_INT16",
    np.int32: "TYPE_INT32",
    np.int64: "TYPE_INT64",
    np.float16: "TYPE_FP16",
    np.float32: "TYPE_FP32",
    np.float64: "TYPE_FP64",
    str: "TYPE_STRING",
    float: "TYPE_FP64",
    int: "TYPE_INT64",
}


model_triton_data_type_maping = {
    bool: "BOOL",
    np.uint8: "UINT8",
    np.uint16: "UINT16",
    np.uint32: "UINT32",
    np.uint64: "UINT64",
    np.int8: "INT8",
    np.int16: "INT16",
    np.int32: "INT32",
    np.int64: "INT64",
    np.float16: "FP16",
    np.float32: "FP32",
    np.float64: "FP64",
    str: "BYTES",
    float: "FP64",
    int: "INT64",
}

try:
    import tensorflow as tf

    model_data_type_maping[tf.bfloat16] = "TYPE_BF16"
    model_triton_data_type_maping[tf.bfloat16] = "BF16"
except ImportError:
    # Code to handle the absence of TensorFlow here
    pass


is_validator = os.environ.get("IS_VALIDATOR")


class ConfigCreationException(Exception):
    def __init__(self, message):
        self.message = message


class TritonPythonModel:
    def initialize(self, args):
        custom_model.##load_func##()

    def execute(self, requests):
        responses = []
        for request in requests:
            required_input = ["##input_list##"]
            inputs = {}
            for each in required_input:
                input_numpy_tensor = pb_utils.get_input_tensor_by_name(
                    request, each["name"]
                )
                if (
                    "optional" in each
                    and each["optional"] == True
                    and input_numpy_tensor is None
                ):
                    continue

                input_numpy_tensor = input_numpy_tensor.as_numpy()
                size = len(input_numpy_tensor)
                shape = input_numpy_tensor.shape
                if shape == (1,):
                    if each["datatype"] == "BYTES":
                        inputs[each["name"]] = np.vectorize(lambda x: x.decode('utf-8'))(input_numpy_tensor)[0]
                    else:
                        inputs[each["name"]] = input_numpy_tensor[0]
                else:
                    input_array = []
                    for i in range(0, size):
                        if each["datatype"] == "BYTES":
                            input_array.append(
                                np.vectorize(lambda x: x.decode('utf-8'))(input_numpy_tensor[i])
                            )
                        else:
                            input_array.append(input_numpy_tensor[i])
                    inputs[each["name"]] = input_array

            inference_outputs = custom_model.##infer_func##(inputs)
            required_output = ["#output_list#"]
            if len(required_output) == 0:
                required_output = []
                try:
                    required_output = TritonPythonModel.create_config_from_json(
                        self, inference_outputs
                    )
                except ValueError as e:
                    raise ConfigCreationException(str(e))
                except Exception as e:
                    raise ConfigCreationException(
                        "Exception occurred while creating output config " + str(e)
                    )
            outputs = []
            for each in required_output:
                if each["name"] in inference_outputs:
                    each_inference_output = inference_outputs[each["name"]]
                    if isinstance(each_inference_output, list):
                        temp_inference_output = []
                        for each_element in each_inference_output:
                            if isinstance(each_element, dict) or isinstance(
                                each_element, tuple
                            ):
                                raise ValueError(
                                    "Output "
                                    + each["name"]
                                    + " is of type dictionary or tuple which is currently not supported. You can try json.dumps("
                                    + each["name"]
                                    + ")"
                                )
                            if each["datatype"] == "BYTES":
                                temp_inference_output.append(
                                    each_element.encode("utf8")
                                )
                            else:
                                temp_inference_output.append(each_element)
                        np_inference_output = np.array(temp_inference_output)
                        outputs.append(
                            pb_utils.Tensor(each["name"], np_inference_output)
                        )
                    elif isinstance(each_inference_output, dict) or isinstance(
                        each_inference_output, tuple
                    ):
                        raise ValueError(
                            "Output "
                            + each["name"]
                            + " is of type dictionary or tuple which is currently not supported. You can try json.dumps("
                            + each["name"]
                            + ")"
                        )
                    else:
                        temp_inference_output = None
                        if each["datatype"] == "BYTES":
                            temp_inference_output = np.array(
                                [each_inference_output.encode("utf8")]
                            )
                        else:
                            temp_inference_output = np.array([each_inference_output])
                        outputs.append(
                            pb_utils.Tensor(each["name"], temp_inference_output)
                        )
                else:
                    raise ValueError(
                        "Requried Output "
                        + each["name"]
                        + " not found in the inference response"
                    )
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)
        return responses

    def finalize(self, args):
        custom_model.##unload_func##(args)

    def create_config_from_json(self, output):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        parent_directory = os.path.dirname(current_directory)
        output_config = []
        output_config_for_return = []
        for key in output:
            each_output = {}
            each_output_for_return = {}
            each_output["name"] = "#" + key + "#"
            each_output_for_return["name"] = key

            type_of_output = TritonPythonModel.get_type(self, output[key])
            if type_of_output == dict or type_of_output == tuple:
                raise ValueError(
                    "Output "
                    + key
                    + " is of type dictionary or tuple which is currently not supported. You can try json.dumps("
                    + key
                    + ")"
                )
            if type_of_output == list:
                raise ValueError(
                    "Output "
                    + key
                    + " is empty list which cannot be used to infer the data type. Please provide a non-empty list as output."
                )
            each_output["data_type"] = TritonPythonModel.map_data_type(
                self, type_of_output
            )
            each_output_for_return["datatype"] = TritonPythonModel.map_triton_data_type(
                self, type_of_output
            )

            shape_of_output = TritonPythonModel.get_shape(self, output[key])
            if len(shape_of_output) > 0:
                each_output["dims"] = shape_of_output
                each_output_for_return["dims"] = shape_of_output
            else:
                each_output["dims"] = [1]
                each_output_for_return["dims"] = [1]

            output_config.append(each_output)
            output_config_for_return.append(each_output_for_return)

        if is_validator == "True":
            config_path = os.path.join(parent_directory, "config.pbtxt")

            with open(config_path, "r") as file:
                content = file.read()

            modified_content = content.replace(
                "output []",
                "output "
                + json.dumps(output_config).replace('"', "").replace("#", '"'),
            )

            with open(config_path, "w", encoding="utf-8") as file:
                file.write(modified_content)

        return output_config_for_return

    def map_data_type(self, variable_type):
        return model_data_type_maping[variable_type]

    def map_triton_data_type(self, variable_type):
        return model_triton_data_type_maping[variable_type]

    def get_shape(self, value):
        shape = []

        if isinstance(value, list):
            shape.append(len(value))
            if shape[0] > 0 and isinstance(value[0], list):
                shape.extend(TritonPythonModel.get_shape(self, value[0]))
        elif isinstance(value, np.ndarray):
            shape.extend(value.shape)

        return shape

    def get_type(self, variable):
        if not isinstance(variable, (list, np.ndarray)):
            return type(variable)
        elif isinstance(variable, (list, np.ndarray)) and len(variable) > 0:
            return TritonPythonModel.get_type(self, variable[0])
        elif isinstance(variable, (list, np.ndarray)) and len(variable) == 0:
            return type([])
        elif not variable:
            return type(None)
        else:
            return TritonPythonModel.get_type(self, variable[0])
