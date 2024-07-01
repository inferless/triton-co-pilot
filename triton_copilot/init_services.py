import os

HOME = os.path.expanduser("~")
CONFIG_FILE_PATH = os.path.join(HOME, ".triton_copilot/config")


def is_initialized():
    return os.path.exists(CONFIG_FILE_PATH)


def create_config_file():
    directory_path = os.path.join(HOME, ".triton_copilot")
    os.makedirs(directory_path)
    #  create an empty config file
    with open(CONFIG_FILE_PATH, "w") as f:
        f.write("")


def save_openai_tokens(openai_key, openai_org):
    # if config file does not exist, create it
    if not is_initialized():
        create_config_file()
    #  save the tokens to the config file
    with open(CONFIG_FILE_PATH, "w") as f:
        f.write(f"OPENAI_KEY={openai_key}\n")
        if openai_org:
            f.write(f"OPENAI_ORG={openai_org}\n")


def save_claude_tokens(claude_api_key):
    # if config file does not exist, create it
    if not is_initialized():
        create_config_file()
    #  save the tokens to the config file
    with open(CONFIG_FILE_PATH, "w") as f:
        f.write(f"CLAUDE_API_KEY={claude_api_key}\n")


def get_model():
    openai_key, _ = get_openai_keys()
    claude_api_key = get_claude_keys()
    if openai_key:
        return "gpt4"
    elif claude_api_key:
        return "claude3"


def get_claude_keys():
    api_key = None
    with open(CONFIG_FILE_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "CLAUDE_API_KEY" in line:
                api_key = line.split("=")[1].strip()

    return api_key


def get_openai_keys():
    api_key = None
    org_id = None
    with open(CONFIG_FILE_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "OPENAI_KEY" in line:
                api_key = line.split("=")[1].strip()
            elif "OPENAI_ORG" in line:
                org_id = line.split("=")[1].strip()

    return api_key, org_id


