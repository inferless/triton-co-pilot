import os
import socket
import subprocess

import anthropic
from openai import OpenAI
import pkg_resources
import shutil

TEMPLATE_DIR = pkg_resources.resource_filename("triton_copilot", "template/")


def copy_code_to_new_dir(src, dest):
    os.system(f"cp -r {src}/* {dest}")


def save_config(output, directory):
    with open(os.path.join(directory, "config.pbtxt"), "w") as f:
        f.write(output)


def cleanup_llm_response(response):
    lines = response.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].startswith("```"):
            lines.pop(i)
        else:
            i += 1
    return "\n".join(lines)


def openai_chat(messages, api_key, org_id):
    client = OpenAI(organization=org_id, api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        temperature=0.5,
        messages=messages,
    )
    return cleanup_llm_response(completion.choices[0].message.content)


def claude_chat(system_message, messages, api_key):
    client = anthropic.Anthropic(
        api_key=api_key,
    )
    message = client.messages.create(
        max_tokens=4096,
        model="claude-3-opus-20240229",
        system=system_message,
        messages=messages
    )
    return cleanup_llm_response(message.content[0].text)


def get_model_name(config_pbtxt):
    lines = config_pbtxt.split("\n")
    # 1st line in config.pbtxt looks like this: name: "gpt-4-turbo-preview"
    model_name = lines[0].replace("\"", "").split(":")[1].strip()
    return model_name


def rename_dir(directory, new_name):
    new_dir_name = "/".join(directory.split("/")[:-1] + [new_name])
    os.rename(directory, new_dir_name)
    return new_dir_name


def cleanup(directory):
    try:
        shutil.rmtree(directory)  # Delete the directory and its contents recursively
    except FileNotFoundError:
        pass


def get_free_ports():
    start = 8000
    end = 8999
    ports = []
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                s.listen(1)
                ports.append(port)
            except OSError:
                continue
        if len(ports) == 3:
            return ports
    raise RuntimeError(f"No free ports found in the range {start}-{end}")


def is_container_running(image_name):
    output = subprocess.run(["docker", "ps"], stdout=subprocess.PIPE)
    return image_name in output.stdout.decode()
