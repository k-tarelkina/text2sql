import re
import os
from tqdm import tqdm
from src.datasets.dataset import Dataset
from src.strategy.example_selection import (
    MaskedQuestionSimilaritySelection,
    QuerySimilaritySelection,
    QuestionSimilaritySelection,
    RandomSelection,
)
from src.strategy.few_shot_text2sql import FewShotText2SQL
from src.strategy.prompt_organization import (
    DAILOrganization,
    FullInformationOrganization,
    SQLOnlyOrganization,
)
from src.llm import LLM
from src.strategy.zero_shot_text2sql import ZeroShotText2SQL
from src.utils.files import write_to_file
from src.utils.log import Logger


def normalize_sql_query(query):
    normalized_query = query.replace("```sql", "")
    normalized_query = normalized_query.replace("```", "")
    normalized_query = normalized_query.replace(" , ", ", ")

    normalized_query = normalized_query.strip()
    normalized_query = normalized_query.rstrip(";")

    normalized_query = re.sub(r"\s+", " ", normalized_query.strip()).lower()

    return normalized_query


def run_prediction(params):
    # extract params
    llm_name = params.get("llm")
    output_folder = params.get("output_folder", "results")
    validation_dataset_limit_rows = params.get("validation_dataset_limit_rows")
    train_dataset_limit_rows = params.get("train_dataset_limit_rows")

    logger = Logger()

    # setup model
    logger.write(f"Start setting up LLM: {llm_name}")
    llm = LLM(llm_name)
    logger.write(f"End setting up LLM")

    # setup dataset
    logger.write(f"Start setting up dataset")
    validation_dataset = Dataset(
        llm, split="validation", limit=validation_dataset_limit_rows
    )
    train_dataset = Dataset(llm, split="train", limit=train_dataset_limit_rows)
    print("validation_dataset", len(validation_dataset))
    print("train_dataset", len(train_dataset))
    logger.write(f"End setting up dataset")

    # get prompt setups
    example_selections = {
        "Random": RandomSelection(llm),
        "QTS": QuestionSimilaritySelection(llm),
        "MQS": MaskedQuestionSimilaritySelection(llm),
        "QRS": QuerySimilaritySelection(llm),
    }
    prompt_organizations = {
        "FI": FullInformationOrganization(),
        "SQL-only": SQLOnlyOrganization(),
        "DAIL": DAILOrganization(),
    }
    n_examples = [1, 2, 4]
    sql_generators = {"Zero-shot": ZeroShotText2SQL(llm, logger)}

    # initiate generators
    logger.write("Start init sql generators")
    for n in n_examples:
        for es_name, es in example_selections.items():
            for po_name, po in prompt_organizations.items():
                sql_generators[f"{es_name}({n}) + {po_name}"] = FewShotText2SQL(
                    llm,
                    dataset=train_dataset,
                    example_selection=es,
                    prompt_organization=po,
                    logger=logger,
                    n_examples=n,
                )
    logger.write("End init sql generators")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # run generations
    for sample in tqdm(validation_dataset):
        for name, generator in sql_generators.items():
            logger.write(f"--------- Start running {name} ---------")

            gold_file_path = os.path.join(output_folder, f"gold_{name}.txt")
            pred_file_path = os.path.join(output_folder, f"pred_{name}.txt")

            answer = generator.generate_sql(sample)

            write_to_file(
                f"{normalize_sql_query(sample['query'])}\t{sample['db_id']}",
                gold_file_path,
            )
            write_to_file(normalize_sql_query(answer), pred_file_path)

            logger.write(f"--------- End running {name} ---------")
