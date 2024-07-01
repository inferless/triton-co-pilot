import ast
import os
import subprocess
import typer
import uuid


def validate_refs(load_ref, infer_ref, unload_ref, file_path):
    if load_ref == "None" or len(load_ref.split(".")) != 2:
        typer.secho("Error: load function reference is missing or incorrect format", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    if infer_ref == "None" or len(infer_ref.split(".")) != 2:
        typer.secho("Error: infer function reference is missing or incorrect format", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()

    if len(unload_ref.split(".")) != 2:
        if unload_ref == "None":
            unload_ref = None
        else:
            typer.secho("Error: unload function reference is incorrect format", fg=typer.colors.BRIGHT_RED)
            raise typer.Exit()

    #  check if the references are present in the file
    with open(file_path, "r") as f:
        data = f.read()
    tree = ast.parse(data)
    for ref in [load_ref, infer_ref, unload_ref]:
        if ref is not None:
            class_name = ref.split(".")[0]
            func_name = ref.split(".")[1]
            found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name == class_name:
                        for child in ast.walk(node):
                            if isinstance(child, ast.FunctionDef):
                                if child.name == func_name:
                                    found = True
                                    break
                if found:
                    break
            if not found:
                raise SystemExit(f"Error: {ref} not found in the file")

    # check if all three references are present in same class or not
    if infer_ref.split(".")[0] != load_ref.split(".")[0]:
        raise SystemExit("Error: infer function should be in the same class as load function")
    if unload_ref is not None and unload_ref.split(".")[0] != load_ref.split(".")[0]:
        raise SystemExit("Error: unload function should be in the same class as load function")


def check_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        if result.returncode == 0:
            typer.echo("GPU is available!")
            return None
        else:
            return "No GPU available, using CPU instead."
    except FileNotFoundError:
        return "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."


def check_docker():
    try:
        result = subprocess.run(['docker', 'info'], stdout=subprocess.PIPE)
        if result.returncode == 0:
            typer.echo("Docker is installed!")
            return None
        else:
            return "Docker is not installed."
    except FileNotFoundError:
        return "Docker command not found. Please install Docker."


def check_dependencies():
    msg = ""
    # check if GPU is available
    err = check_gpu()
    if err:
        msg = err
    # check if docker is installed
    err = check_docker()
    if err:
        msg += "\n" + err

    if msg != "":
        return msg
    return None


def validate_file(file_path):
    try:
        with open(file_path, "r") as f:
            data = f.read()
        ast.parse(data)
        if file_path.split("/")[-1] == "model.py":
            rand_id = uuid.uuid4().hex[:4]
            new_file_path = file_path.replace("model.py", f"model_{rand_id}.py")
            os.rename(file_path, new_file_path)
            return new_file_path
        # check if there is a file called model.py in the directory
        if "model.py" in os.listdir(os.path.dirname(file_path)):
            typer.secho("Error: model.py file found in the directory, please do not use `model.py` as "
                        "module name as it is reserved for triton entrypoint ", fg=typer.colors.BRIGHT_RED)
            raise typer.Exit()
        return file_path
    except IsADirectoryError:
        typer.secho("Error: Path is a directory", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except FileNotFoundError:
        typer.secho("Error: File not found", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except SyntaxError as e:
        typer.secho(f"Syntax Error: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
