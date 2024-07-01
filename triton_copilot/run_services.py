import time
import requests
import typer
import subprocess

def run_docker_image(tag, volume, env_vars):
    try:
        command = f"docker run --rm -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002"
        if volume is not None:
            command = f"{command} -v {volume}"
        if env_vars:
            env_vars_str = " ".join(f"-e {key}={value}" for key, value in env_vars.items())
            command = f"{command} {env_vars_str}"
        command = f"{command} {tag}"
        subprocess.run(
            command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}, error: {}".format(e.cmd, e.returncode, e.output, e.stderr))
    except Exception as e:
        raise SystemExit(f"Error: {e}")


def wait_for_container_to_start():
    url = "http://localhost:8000/v2/health/ready"
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


def get_curl_command():
    url = "http://localhost:8000/v2/repository/index"
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
                typer.secho(f"curl -X POST http://localhost:8000/v2/models/{model_name}/versions/{model_version}/"
                            f"infer -H 'Content-Type: application/json' -d <payload>", fg=typer.colors.BRIGHT_GREEN)
                typer.secho("Replace <payload> with the actual payload, refer triton docs for payload format https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_binary_data.md", fg=typer.colors.YELLOW)
            else:
                typer.secho(f"Model {model.get('name')} is not ready", fg=typer.colors.BRIGHT_YELLOW)
    else:
        typer.secho("Failed to get model details", fg=typer.colors.BRIGHT_RED)
        typer.Exit()
