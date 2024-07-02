import time
import requests
import typer
import subprocess
import json
from triton_copilot.utils import get_free_ports, is_container_running

def run_docker_image(tag, volume, env_vars):
    try:
        ports = get_free_ports()
        command = f"docker run --rm -d --gpus all -p {ports[0]}:8000 -p {ports[1]}:8001 -p {ports[2]}:8002"
        if volume is not None:
            command = f"{command} -v {volume}"
        if env_vars:
            env_vars_str = " ".join(f"-e {key}={value}" for key, value in env_vars.items())
            command = f"{command} {env_vars_str}"
        command = f"{command} {tag}"
        completed_process = subprocess.run(
            command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        build_output_lines = completed_process.stderr.decode().split('\n')
        time.sleep(10)
        if is_container_running(tag):
            return ports[0]
        else:
            typer.secho("\nFailed to start Triton Inference Server, Build logs:\n", fg=typer.colors.BRIGHT_RED)
            for line in build_output_lines:
                typer.secho(line)
            raise RuntimeError()
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}, error: {}".format(e.cmd, e.returncode, e.output, e.stderr))


def wait_for_container_to_start(port):
    url = f"http://localhost:{port}/v2/health/ready"
    wait_time = 300
    while wait_time > 0:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
            else:
                time.sleep(10)
                wait_time -= 10
        except Exception as e:
            time.sleep(10)
            wait_time -= 10

    return False


def get_curl_command(port):
    url = f"http://localhost:{port}/v2/repository/index"
    response = requests.post(url)
    if response.status_code == 200:
        data = response.json()
        if len(data) == 0:
            typer.secho("No models found", fg=typer.colors.BRIGHT_RED)
            typer.Exit()
        for model in data:
            if model.get("state") == "READY":
                model_name = model.get("name")
                model_version = model.get("version")
                typer.secho(f"Model name: {model_name}, Model version: {model_version} is ready, \n curl command to infer:")
                typer.secho(f"curl -X POST http://localhost:{port}/v2/models/{model_name}/versions/{model_version}/"
                            f"infer -H 'Content-Type: application/json' -d <payload>", fg=typer.colors.BRIGHT_GREEN)
                typer.secho("Replace <payload> with the actual payload!! \nsample payload was provided as part of the build step", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"Model {model.get('name')} is not ready", fg=typer.colors.BRIGHT_YELLOW)
    else:
        typer.secho("Failed to get model details", fg=typer.colors.BRIGHT_RED)
        typer.Exit()


def echo_run_instructions(tag, triton_inputs, model_name):
    typer.secho(f"Command to run the Triton Inference Server:", fg=typer.colors.BRIGHT_GREEN)
    typer.echo(f"triton-copilot run {tag}")

    typer.secho(f"Curl Command to infer:", fg=typer.colors.BRIGHT_GREEN)
    typer.echo(f"curl -X POST http://localhost:8000/v2/models/{model_name}/versions/1/infer -H 'Content-Type: application/json' -d '{json.dumps(triton_inputs)}'")
