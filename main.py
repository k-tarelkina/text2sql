import click
from src.generate_predictions import run_prediction
from src.generate_evaluation import run_evaluation
from src.utils.cli import params_path
from src.utils.files import read_yaml
from src.utils.log import Logger


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
    pred_params = params.get("prediction")

    logger = Logger("prediction")
    logger.write(f"Start prediction task with params: {pred_params}")
    run_prediction(pred_params, logger)
    logger.write("Prediction is finished")


@main.command()
@params_path
@click.option(
    "--params_path",
    help="The path to the params yaml file",
    default="params.yaml",
    type=str,
)
def evaluate(params_path: str):
    params = read_yaml(params_path)
    eval_params = params.get("evaluation")

    logger = Logger("evaluation")
    logger.write(f"Start evaluation task with params: {eval_params}")
    run_evaluation(eval_params, logger)
    logger.write("Evaluation is finished")


if __name__ == "__main__":
    main()
