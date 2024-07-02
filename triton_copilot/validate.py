import ast
import subprocess
import typer


def validate_refs(load_ref, infer_ref, unload_ref, file_path):
    try:
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
    except Exception as e:
        typer.secho(f"Failed to validate references: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()


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
    except IsADirectoryError:
        typer.secho("Error: Path is a directory", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except FileNotFoundError:
        typer.secho("Error: File not found", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
    except SyntaxError as e:
        typer.secho(f"Syntax Error: {e}", fg=typer.colors.BRIGHT_RED)
        raise typer.Exit()
