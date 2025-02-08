import click
from generate_predictions import run_prediction
from utils.cli import params_path
from utils.files import read_yaml


@click.group()
def main():
    pass


@main.command()
@params_path
@click.option(
    "--params_path",
    help="The path to the params yaml file",
    default="params.yaml",
    type=str,
)
def predict(params_path: str):
    params = read_yaml(params_path)
    run_prediction(params.get("prediction"))


if __name__ == "__main__":
    main()
