

# Triton-Server Co-Pilot

Triton-Server Co-Pilot is an innovative tool designed to streamline the process of converting any model code into Triton-Server compatible code, simplifying deployment on NVIDIA Triton Inference Server. This project automatically generates necessary configuration files (`config.pbtxt`) and custom wrapper code (`model.py`), among others, facilitating seamless integration and deployment of AI models in production environments.

## Features and Benefits

- **Automatic Code Conversion**: Convert your model code into Triton-Server compatible code.
- **Configuration File Generation**: Automatically generates `config.pbtxt` files tailored to your specific model requirements.

## Installation
`pip install triton-copilot`

## Initialization
This cli too requires api keys to either openai's gpt4 model or claude3 model
| **Usage**                       |
|---------------------------------|
| `triton-copilot init [OPTIONS]` |

The options are prompted when you run the command
```
Select the model you want to build Triton image for [gpt4, claude3]: gpt4
Enter your OpenAI API key: ""
Enter your OpenAI organization ID [None]: ""
```

## Build Triton Server
`triton-copilot build`
This command builds the triton docker image by making necessary modifications to the code using genAi

* This command will prompt for Options if not provided, to avoid prompting use --no-prompt *

| **Usage**                               |
|----------------------------------------|
| `triton-copilot build [OPTIONS] FILE_PATH` |

| **Arguments** | **Type** | **Description** | **Default** | **Required** |
|---------------|----------|-----------------|-------------|--------------|
| `file_path`   | TEXT     | Path to the python file | None | Yes |

| **Options**                   | **Type** | **Description**                         | **Default** | **Required** |
|-------------------------------|----------|-----------------------------------------|-------------|--------------|
| `--tag`                       | TEXT     | Docker image tag                        | None        | Yes          |
| `--triton-version`            | TEXT     | Triton version                          | 24.05-py3   | No           |
| `--load-ref`                  | TEXT     | Reference to the load function [Class.Function]          | None        | No           |
| `--infer-ref`                 | TEXT     | Reference to the inference function[Class.Function]     | None        | No           |
| `--unload-ref`                | TEXT     | Reference to the unload function [Class.Function]       | None        | No           |
| `--sample-input-file`         | TEXT     | File path to sample input payload       | None        | No           |
| `--sample-output-file`        | TEXT     | File path to sample output payload      | None        | No           |
| `--max-batch-size`            | INTEGER  | Max batch size                          | None        | No           |
| `--preferred-batch-size`      | INTEGER  | Preferred batch size                    | None        | No           |
| `--no-prompt`                 | BOOL     | Skip prompts                            | no-no-prompt| No           |
| `--help`                      | -        | Show this message and exit              | -           | -            |


## Run Triton Server
`triton-copilot run`
This command runs the triton docker image in detached mode

Usage: triton-copilot run [OPTIONS] TAG

**Arguments**

| Argument | Type | Description             | Default | Required |
|----------|------|-------------------------|---------|----------|
| `tag`    | TEXT | Docker image tag        | None    | Yes      |

**Options**

| Option     | Type | Description                        | Default | Format           |
|------------|------|------------------------------------|---------|------------------|
| `--volume` | TEXT | Volume name                        | None    |source:destination|
| `--env`    | TEXT | Environment variables as a json    | None    |'{"KEY": "VALUE}' |
| `--help`   | -    | Show this message and exit         | -       |                  |


## Prerequisites
- Your model code is ready for conversion

## Contributing & Support

- [GitHub issues](https://github.com/inferless/triton-copilot/issues/new) \(Bug reports, feature requests, and anything roadmap related)
