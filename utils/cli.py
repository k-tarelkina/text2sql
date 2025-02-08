import click


def params_path(func):
    return click.option(
        "--params_path",
        default="params.yaml",
        help="The path to the yaml file with params",
        type=str,
    )(func)
