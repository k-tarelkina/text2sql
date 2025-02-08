import yaml


def read_yaml(file_path):
    """Read content of yaml file"""
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def write_to_file(text, file_path):
    """Append text to the file"""
    with open(file_path, "a") as file:
        file.write(text + "\n")
