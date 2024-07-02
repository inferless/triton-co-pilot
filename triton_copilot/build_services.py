import shutil
import typer
import ast
import subprocess
import os


def get_function_refs(file_path):
    load_ref, infer_ref, unload_ref = infer_function_refs(file_path)
    if load_ref is not None:
        load_ref = typer.prompt("Enter the load function reference", default=load_ref)
    else:
        load_ref = typer.prompt("Enter the load function reference [Ex: TritonClass.load]")

    if infer_ref is not None:
        infer_ref = typer.prompt("Enter the infer function reference", default=infer_ref)
    else:
        infer_ref = typer.prompt("Enter the infer function reference [Ex: TritonClass.infer]")

    if unload_ref is not None:
        unload_ref = typer.prompt("Enter the unload function reference", default=unload_ref)
    else:
        unload_ref = typer.prompt("Enter the unload function reference [Ex: TritonClass.unload]", default="None")

    return load_ref, infer_ref, unload_ref


def infer_function_refs(file_path):
    with open(file_path, "r") as f:
        data = f.read()
    tree = ast.parse(data)
    load_ref = None
    infer_ref = None
    unload_ref = None
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "load":
                load_ref = f"{node.parent.name}.{node.name}"
            elif node.name == "infer":
                infer_ref = f"{node.parent.name}.{node.name}"
            elif node.name == "unload":
                unload_ref = f"{node.parent.name}.{node.name}"
    return load_ref, infer_ref, unload_ref


def build_image(project_path, tag, triton_version, requirements_file):
    try:
        if requirements_file != "None":
            shutil.copy(requirements_file, f"{project_path}/1/requirements.txt")
        file = open(f"{project_path}/Dockerfile", "r")
        lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i].replace("##tritonversion##", triton_version)
            lines[i] = lines[i].replace('##model_name##', project_path.split('/')[-1])

        file = open(f"{project_path}/Dockerfile", "w")
        file.writelines(lines)
        file.close()

        # check requirement.txt file exists
        if not os.path.exists(f"{project_path}/1/requirements.txt"):
            # create an empty requirements.txt file
            with open(f"{project_path}/1/requirements.txt", "w") as f:
                f.write("")

        docker_build_cmd = "docker build -t {tag} {dir} -f {dir}/Dockerfile".format(
            dir=project_path, tag=tag
        )
        completed_process = subprocess.run(
            docker_build_cmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}, error: {}".format(e.cmd, e.returncode, e.output, e.stderr))
    except Exception as e:
        raise SystemExit(f"Error: {e}")
