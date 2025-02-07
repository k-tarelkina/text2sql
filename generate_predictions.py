from dataset import Dataset
from example_selection import (
    MaskedQuestionSimilaritySelection,
    QuerySimilaritySelection,
    QuestionSimilaritySelection,
    RandomSelection,
)
from few_shot_text2sql import FewShotText2SQL
from llm import LLM
from prompt_organization import (
    DAILOrganization,
    FullInformationOrganization,
    SQLOnlyOrganization,
)
from zero_shot_text2sql import ZeroShotText2SQL
import re
import os
from tqdm import tqdm

######### TODO make as CLI arguments
validation_dataset_limit_rows = 100
train_dataset_limit_rows = 100
output_folder = "results"
llm_name = "mistralai/Ministral-8B-Instruct-2410"
#########


def write_to_file(text, file_path):
    with open(file_path, "a") as file:
        file.write(text + "\n")


def normalize_sql_query(query):
    normalized_query = query.replace("```sql", "")
    normalized_query = normalized_query.replace("```", "")
    normalized_query = normalized_query.replace(" , ", ", ")

    normalized_query = normalized_query.strip()
    normalized_query = normalized_query.rstrip(";")

    normalized_query = re.sub(r"\s+", " ", normalized_query.strip()).lower()

    return normalized_query


def main():
    llm = LLM(llm_name)

    validation_dataset = Dataset(
        llm, split="validation", limit=validation_dataset_limit_rows
    )
    train_dataset = Dataset(llm, split="train", limit=train_dataset_limit_rows)

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

    n_examples = [2, 4, 8]

    sql_generators = {"Zero-shot": ZeroShotText2SQL(llm)}

    for n in n_examples:
        for es_name, es in example_selections.items():
            for po_name, po in prompt_organizations.items():
                sql_generators[f"{es_name}({n}) + {po_name}"] = FewShotText2SQL(
                    llm,
                    dataset=train_dataset,
                    example_selection=es,
                    prompt_organization=po,
                    n_examples=n,
                )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for sample in tqdm(validation_dataset):
        for name, generator in sql_generators.items():
            gold_file_path = os.path.join(output_folder, f"gold_{name}.txt")
            pred_file_path = os.path.join(output_folder, f"pred_{name}.txt")

            answer = generator.generate_sql(sample)

            write_to_file(
                f"{normalize_sql_query(sample['query'])}\t{sample['db_id']}",
                gold_file_path,
            )
            write_to_file(normalize_sql_query(answer), pred_file_path)


if __name__ == "__main__":
    main()
