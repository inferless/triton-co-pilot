import os
from triton_copilot.init_services import save_openai_tokens, save_claude_tokens, is_initialized, get_model
from triton_copilot.run_services import wait_for_container_to_start, get_curl_command, run_docker_image, \
    echo_run_instructions
from triton_copilot.utils import rename_dir, cleanup
from triton_copilot.triton_services import create_triton_code, create_config_pbtxt, modify_modelpy, get_triton_inputs
from triton_copilot.build_services import build_image, get_function_refs
from triton_copilot.validate import validate_refs, check_dependencies, validate_file

import typer
from enum import Enum
from rich.progress import Progress, SpinnerColumn, TextColumn
import json
app = typer.Typer()


class ModelChoices(str, Enum):
    gpt4 = "gpt4"
    claude3 = "claude3"


@app.command(name="init")
def initialize():
    try:
        typer.secho("Welcome to Triton Copilot!", fg=typer.colors.CYAN)
        typer.secho("This tool helps you build and run Triton Inference Server images. Let's get started!", fg=typer.colors.CYAN)
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
        ) as progress:
            progress.add_task("Checking dependencies", total=None)
            error = check_dependencies()
            if error:
                typer.secho(f"Please install the missing dependencies and try again. {error}", fg=typer.colors.BRIGHT_RED)
                # raise typer.Exit()
            typer.echo("Dependencies validated successfully\n")

        typer.secho("We support OpenAI's gpt-4-turbo-preview, Anthropic's claude-3-opus-20240229 models to generate "
                    "triton compatible code.", fg=typer.colors.MAGENTA)
        model = typer.prompt("Select the model you want to build Triton image for [gpt4, claude3]", type=ModelChoices)
        if model == "gpt4":
            openai_key = typer.prompt("Enter your OpenAI API key")
            openai_org = typer.prompt("Enter your OpenAI organization ID", default="None")
            if openai_org == "None":
                openai_org = None
            save_openai_tokens(openai_key, openai_org)
        elif model == "claude3":
            claude_api_key = typer.prompt("Enter your Claude API key")
            save_claude_tokens(claude_api_key)

        typer.echo("You are all set to build Triton Server image")
    except Exception as e:
        typer.secho(f"Something went wrong: {e}", fg=typer.colors.BRIGHT_RED)


@app.command(name="build")
def build_triton(
        file_path: str = typer.Argument(help="Path to the python file"),
        tag: str = typer.Option(help="Docker image tag", prompt="Enter the image tag"),
        triton_version: str = typer.Option("24.05-trtllm-python-py3", help="Triton version", prompt="Enter the Triton version"),
        load_ref: str = typer.Option(default="None", help="Reference to the load function"),
        infer_ref: str = typer.Option(default="None", help="Reference to the inference function"),
        unload_ref: str = typer.Option(default="None", help="Reference to the unload function"),
        sample_input_file: str = typer.Option("None", help="file path to sample input payload"),
        sample_output_file: str = typer.Option("None", help="file path to Sample output payload"),
        requirements_file: str = typer.Option("None", help="file path to requirements.txt"),
        max_batch_size: int = typer.Option(None, help="Max batch size"),
        preferred_batch_size: int = typer.Option(None, help="Preferred batch size"),
        no_prompt: bool = typer.Option(False, help="Skip prompts"),
):
    try:
        if not is_initialized():
            typer.secho("Please run 'triton-copilot init' to initialize the tool", fg=typer.colors.BRIGHT_RED)
            typer.Exit()
        validate_file(file_path)
        if not no_prompt:
            if load_ref == "None":
                load_ref, infer_ref, unload_ref = get_function_refs(file_path)
            if sample_input_file == "None":
                sample_input_file = typer.prompt("File path of sample input payload", default="None")
            if sample_output_file == "None":
                sample_output_file = typer.prompt("File path of sample output payload", default="None")
            if requirements_file == "None":
                directory = "." if os.path.dirname(file_path)=="" else os.path.dirname(file_path)
                if "requirements.txt" in os.listdir(directory):
                    requirements_file = typer.prompt("File path of requirements.txt", default="requirements.txt")
                else:
                    requirements_file = typer.prompt("File path of requirements.txt", default="None")
            if not max_batch_size:
                batching = typer.confirm("Do you want to enable batching?")
                max_batch_size = None
                preferred_batch_size = None
                if batching:
                    max_batch_size = typer.prompt("Enter the max batch size", default=2)
                    preferred_batch_size = typer.prompt("Enter the preferred batch size", default=2)
        validate_refs(load_ref, infer_ref, unload_ref, file_path)
        model = get_model()
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
        ) as progress:
            progress.add_task("Generating Triton code\n", total=None)
            project_path = create_triton_code(file_path, load_ref, infer_ref, unload_ref)
            model_name = create_config_pbtxt(model, file_path, project_path, sample_input_file, sample_output_file, max_batch_size, preferred_batch_size)
            inputs, outputs = modify_modelpy(project_path, model)
            triton_inputs = get_triton_inputs(model, inputs)
            temp_path = project_path
            project_path = rename_dir(project_path, model_name)
            progress.add_task("Building image\n", total=None)
            build_image(project_path, tag, triton_version, requirements_file)
            cleanup(temp_path)
            cleanup(project_path)
        typer.echo(f"Image built successfully with tag: {tag}")
        echo_run_instructions(tag, triton_inputs, model_name)
    except Exception as e:
        typer.secho(f"Something went wrong: {e}", fg=typer.colors.BRIGHT_RED)
        cleanup(temp_path) if "temp_path" in locals() else None
        cleanup(project_path) if "project_path" in locals() else None
    except KeyboardInterrupt or SystemExit:
        typer.secho("Exiting...", fg=typer.colors.BRIGHT_RED)
        cleanup(temp_path) if "temp_path" in locals() else None
        cleanup(project_path) if "project_path" in locals() else None


@app.command(name="run")
def run_triton_image(
        tag: str = typer.Argument(help="Docker image tag"),
        volume: str = typer.Option("None", help="Volume name"),
        env: str = typer.Option("None", help="Environment variables as a json"),
):
    try:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
        ) as progress:
            if not is_initialized():
                typer.secho("Please run 'triton-copilot init' to initialize the tool", fg=typer.colors.BRIGHT_RED)
                typer.Exit()
            if volume == "None":
                volume = None
            if env == "None":
                env_vars = None
            else:
                env_vars = json.loads(env)
            task_id = progress.add_task("Running Triton Inference Server", total=None)
            port = run_docker_image(tag, volume, env_vars)
            progress.stop_task(task_id)
            task_id = progress.add_task("Waiting for Triton Inference Server to start", total=None)
            is_started = wait_for_container_to_start(port)
            progress.stop_task(task_id)
            if is_started:
                typer.echo("Triton Inference Server is running")
                get_curl_command(port)
            else:
                typer.secho("Triton Inference Server failed to start in 300s, please check docker logs",
                            fg=typer.colors.BRIGHT_RED)
    except Exception as e:
        typer.secho(f"Something went wrong: {e}", fg=typer.colors.BRIGHT_RED)


if __name__ == "__main__":
    app()
