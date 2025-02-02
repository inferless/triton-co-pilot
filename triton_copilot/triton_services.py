import os
import typer
from triton_copilot.constants import system_prompt_config_pbtxt
from triton_copilot.init_services import get_claude_keys, get_openai_keys
from triton_copilot.utils import copy_code_to_new_dir, claude_chat, openai_chat, get_model_name, save_config, cleanup
from triton_copilot.validate import validate_file
import pkg_resources
import ast
import uuid

TEMPLATE_DIR = pkg_resources.resource_filename("triton_copilot", "template/")


def get_user_prompt(file_path, sample_input, sample_output):
    with open(file_path, "r") as f:
        app_code = f.read()
    user_prompt_text = f"Provided Code Snippet: \n {app_code}"
    if sample_input:
        user_prompt_text += f"\nUser has provided a sample_input. The contents are: \n {sample_input}"
    elif sample_output:
        user_prompt_text += f"\nUser has provided a sample_output. The contents are: \n {sample_output}"

    user_prompt = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt_text
                }
            ]
        }
    ]
    user_prompt.append(
        {
            "role": "assistant",
            "content": "The config.pbtxt is:"
        }
    )
    return user_prompt


def map_input_output_data_types(input_output):
    model_triton_data_type_maping = {
        "TYPE_BOOL": "BOOL",
        "TYPE_UINT8": "UINT8",
        "TYPE_UINT16": "UINT16",
        "TYPE_UINT32": "UINT32",
        "TYPE_UINT64": "UINT64",
        "TYPE_INT8": "INT8",
        "TYPE_INT16": "INT16",
        "TYPE_INT32": "INT32",
        "TYPE_INT64": "INT64",
        "TYPE_FP16": "FP16",
        "TYPE_FP32": "FP32",
        "TYPE_FP64": "FP64",
        "TYPE_STRING": "BYTES",
        "TYPE_BF16": "BF16",
    }
    for input in input_output["inputs"]:
        input["data_type"] = model_triton_data_type_maping[input["data_type"]]
    for output in input_output["outputs"]:
        output["data_type"] = model_triton_data_type_maping[output["data_type"]]

    return input_output["inputs"], input_output["outputs"]


def get_input_output(file_path, model):
    with open(file_path, "r") as f:
        data = f.read()

    if model == "gpt4":
        messages = [
            {
                "role": "system",
                "content": """
                    Extract the input and output as dictionaries from the provided text.
                    The output should be a dictionary with the keys "inputs" and "outputs".
                    Both input and output should be a list of dictionaries
                """
            },
            {
                "role": "user",
                "content": data
            }
        ]
        api_key, org_id = get_openai_keys()
        input_output = openai_chat(messages, api_key, org_id)
        input_output = ast.literal_eval(input_output)
    elif model == "claude3":
        system_message = """
            Extract the input and output as dictionaries from the provided text.
            The output should be a dictionary with the keys "inputs" and "outputs".
            Both input and output should be a list of dictionaries
        """
        messages = [
            {
                "role": "user",
                "content": data
            },
            {
                "role": "assistant",
                "content": "The extracted input and output is:"
            }
        ]
        api_key = get_claude_keys()
        input_output = claude_chat(system_message, messages, api_key)
        input_output = ast.literal_eval(input_output)

    return map_input_output_data_types(input_output)


def get_triton_inputs(model, inputs):
    try:
        if model == "gpt4":
            messages = [
                {
                    "role": "system",
                    "content": """
                        Generate sample triton inputs payload for given schema. 
                        Return a dictionary only with "inputs" as the key.
                        This object should be a valid python dictionary without any comments or extra text.
                        Use only doule quotes in the response, do not use single quotes at all
                    """
                },
                {
                    "role": "user",
                    "content": str(inputs)
                }
            ]
            api_key, org_id = get_openai_keys()
            triton_inputs = openai_chat(messages, api_key, org_id)
            triton_inputs = ast.literal_eval(triton_inputs)
        elif model == "claude3":
            system_message = """
                Generate sample triton inputs payload for given schema. 
                Return a dictionary only with "inputs" as the key.
                This object should be a valid python dictionary without any comments or extra text.
                Use only doule quotes in the response, do not use single quotes at all
            """
            messages = [
                {
                    "role": "user",
                    "content": str(inputs)
                },
                {
                    "role": "assistant",
                    "content": "The extracted input is:"
                }
            ]
            api_key = get_claude_keys()
            triton_inputs = claude_chat(system_message, messages, api_key)
            triton_inputs = ast.literal_eval(triton_inputs)

        return triton_inputs
    except ValueError as e:
        typer.secho(f"generated triton inputs are not in the correct format, try again: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except SyntaxError as e:
        typer.secho(f"generated triton inputs are not in the correct format, try again: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except Exception as e:
        typer.secho(f"Failed to get triton inputs: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()


def modify_modelpy(directory, model):
    try:
        inputs, outputs = get_input_output(f"{directory}/config.pbtxt", model)
        for each in inputs:
            each["datatype"] = each.pop("data_type")
        for each in outputs:
            each["datatype"] = each.pop("data_type")

        file = open(f"{directory}/1/model.py", "r")
        lines = file.readlines()
        file.close()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('["##input_list##"]', str(inputs))
            lines[i] = lines[i].replace('["#output_list#"]', str(outputs))

        file = open(f"{directory}/1/model.py", "w")
        file.writelines(lines)

        return inputs, outputs
    except Exception as e:
        typer.secho(f"Failed at creating model.py: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()


def create_triton_code(file_path, load_ref, infer_ref, unload_ref):
    try:
        base_dir = os.path.dirname(file_path)
        if base_dir == "":
            base_dir = "."
        os.makedirs("./temp/1", exist_ok=True)
        temp_dir = "./temp"
        copy_code_to_new_dir(base_dir, temp_dir + "/1")
        if file_path.split("/")[-1] == "model.py":
            rand_id = uuid.uuid4().hex[:4]
            file_path = f"{temp_dir}/1/model_{rand_id}.py"
            os.rename(f"{temp_dir}/1/model.py", file_path)

        # check if there is a file called model.py in the directory
        if "model.py" in os.listdir(os.path.dirname(temp_dir + "/1")):
            typer.secho("Error: model.py file found in the directory, please do not use `model.py` as "
                        "module name as it is reserved for triton entrypoint ", fg=typer.colors.BRIGHT_RED)
            raise typer.Exit()

        copy_code_to_new_dir(TEMPLATE_DIR, temp_dir)
        cleanup(f"{temp_dir}/1/temp")
        file = open(f"{temp_dir}/1/model.py", "r")
        lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("##model_class##", load_ref.split(".")[0])
            lines[i] = lines[i].replace("##load_func##", load_ref.split(".")[1])
            lines[i] = lines[i].replace("##infer_func##", infer_ref.split(".")[1])
            lines[i] = lines[i].replace("##file##", file_path.split("/")[-1].split(".")[0])
            if unload_ref is not None and unload_ref != "None":
                lines[i] = lines[i].replace("##unload_func##", unload_ref.split(".")[1])

        if unload_ref is None or unload_ref == "None":
            lines = [line for line in lines if "finalize" not in line and "unload" not in line]

        file = open(f"{temp_dir}/1/model.py", "w")
        file.writelines(lines)
        return temp_dir
    except Exception as e:
        typer.secho(f"Failed to create triton code: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()


def create_config_pbtxt(model, file_path, project_path, sample_input_file, sample_output_file, max_batch_size, preferred_batch_size):
    try:
        if sample_input_file == "None":
            sample_input = None
        else:
            validate_file(sample_input_file)
            with open(sample_input_file, "r") as f:
                sample_input = f.read()
        if sample_output_file == "None":
            sample_output = None
        else:
            validate_file(sample_output_file)
            with open(sample_output_file, "r") as f:
                sample_output = f.read()
        output = ""
        if model == "gpt4":
            api_key, org_id = get_openai_keys()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_config_pbtxt
                }
            ]
            user_prompt = get_user_prompt(file_path, sample_input, sample_output)
            messages.extend(user_prompt)
            output = openai_chat(messages, api_key, org_id)
        elif model == "claude3":
            api_key = get_claude_keys()
            new_dir, user_prompt = get_user_prompt(file_path, sample_input, sample_output)
            output = claude_chat(system_prompt_config_pbtxt, user_prompt, api_key)

        if max_batch_size is not None:
            max_queue_delay = 500000
            output = output + "\ndynamic_batching {\n"
            output = output + f"  preferred_batch_size: [ {preferred_batch_size} ]\n"
            output = output + f"  max_queue_delay_microseconds: {max_queue_delay}\n"
            output = output + "}\n"
            output = output + f"max_batch_size: {max_batch_size}\n"

        save_config(output, project_path)
        return get_model_name(output)
    except FileNotFoundError:
        typer.secho("Error: Sample input or output file not found", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except Exception as e:
        typer.secho(f"Failed to create config.pbtxt: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
