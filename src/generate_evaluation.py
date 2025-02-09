import gdown
import os
import zipfile
from src.evaluation.evaluation import build_foreign_key_map_from_json, evaluate


def download_from_gdrive(file_id, output_folder, logger):
    extracted_folder = os.path.join(output_folder, "testsuitedatabases")
    if os.path.exists(extracted_folder):
        logger.write("Extraction of test database already done, skipping download")
        return

    zip_path = os.path.join(output_folder, "testsuitedatabases.zip")
    if not os.path.exists(zip_path):
        logger.write(f"Downloading file from google drive: {file_id}")
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False
        )
    else:
        logger.write(f"Zip file of test database already exists, skipping download")

    logger.write(f"Extracting {zip_path} to {output_folder}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_folder)

    logger.write("Download and extraction completed")


def run_evaluation(params, logger):
    predictions_folder = params.get("predictions_folder")
    results_folder = params.get("output_folder")
    db_dir = params.get("db_dir")
    table = params.get("table")
    etype = params.get("etype")
    assert etype in ["all", "exec", "match", "time"], "Unknown evaluation method"

    os.makedirs(results_folder, exist_ok=True)

    download_from_gdrive("1mkCx2GOFIqNesD4y8TDAO1yX1QZORP5w", "", logger)

    files = os.listdir(predictions_folder)

    gold_files = [f for f in files if f.startswith("gold_")]
    pred_files = [f for f in files if f.startswith("pred_")]

    for gold_file in gold_files:
        for pred_file in pred_files:
            gold_method_name = gold_file[5:]
            pred_method_name = pred_file[5:]

            # get files pred and gold files with the same prediction methods
            if gold_method_name == pred_method_name:
                logger.write(f"Start evaluating {gold_method_name}")

                gold_path = os.path.join(predictions_folder, gold_file)
                pred_path = os.path.join(predictions_folder, pred_file)
                result_file = os.path.join(
                    results_folder, f"results_{gold_method_name}"
                )

                kmaps = build_foreign_key_map_from_json(table)
                logger.write(f"kmaps {kmaps}")
                evaluate(gold_path, pred_path, db_dir, etype, kmaps, result_file)

                logger.write(f"End evaluating {gold_method_name}")
