from abc import abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from database_catalog import DATABASE_CATALOG
from llm import LLM
import numpy as np
import torch
import random

SQL_KEYWORDS = [
    "SELECT",
    "FROM",
    "WHERE",
    "INSERT",
    "INTO",
    "VALUES",
    "UPDATE",
    "DELETE",
    "JOIN",
    "INNER",
    "LEFT",
    "RIGHT",
    "FULL",
    "OUTER",
    "ON",
    "GROUP",
    "BY",
    "ORDER",
    "HAVING",
    "DISTINCT",
    "COUNT",
    "SUM",
    "AVG",
    "MIN",
    "MAX",
    "AS",
    "AND",
    "OR",
    "NOT",
    "IN",
    "LIKE",
    "LIMIT",
    "OFFSET",
    "BETWEEN",
    "CASE",
    "WHEN",
    "THEN",
    "END",
    "ELSE",
    "ASC",
    "DESC",
    "UNION",
    "ALL",
    "EXISTS",
    "INTERSECT",
    "EXCEPT",
    "CREATE",
    "TABLE",
    "DROP",
    "ALTER",
    "ADD",
    "PRIMARY",
    "KEY",
    "FOREIGN",
    "CONSTRAINT",
    "DEFAULT",
    "CHECK",
    "INDEX",
    "VIEW",
    "TRIGGER",
    "PROCEDURE",
    "FUNCTION",
    "CAST",
]


class ExampleSelection:
    def __init__(self, llm: LLM):
        self.llm = llm

    def compute_similarity(example_a, example_b, field):
        return cosine_similarity([example_a[field]], [example_b[field]])[0][0]

    @abstractmethod
    def select_examples(self, sample, dataset, n_examples):
        pass


class RandomSelection(ExampleSelection):
    def select_examples(self, sample, dataset, n_examples):
        return random.sample(dataset, n_examples)


class QuestionSimilaritySelection(ExampleSelection):
    def select_examples(self, sample, dataset, n_examples):
        cos_sims = np.array([])

        for candidate in dataset:
            cos_sims = np.append(
                cos_sims, self.compute_similarity(sample, candidate, "question_vector")
            )

        sorted_indices = np.argsort(cos_sims)[::-1]
        return np.array(dataset)[sorted_indices][:n_examples].tolist()


class MaskedQuestionSimilaritySelection(ExampleSelection):
    def select_examples(self, sample, dataset, n_examples):
        cos_sims = np.array([])

        for candidate in dataset:
            cos_sims = np.append(
                cos_sims,
                self.compute_similarity(sample, candidate, "masked_question_vector"),
            )

        sorted_indices = np.argsort(cos_sims)[::-1]
        return np.array(dataset)[sorted_indices][:n_examples].tolist()


class QuerySimilaritySelection(ExampleSelection):
    def __generate_sql_query(self, question, database_schema):
        prompt = f"Question: {question}\nSchema: {database_schema}\nGenerate a SQL query for the above question based on the schema and with no explanation "
        return self.llm.answer(prompt)

    def __encode_query_to_binary_vector(self, query):
        query = query.upper()
        return np.array([1 if keyword in query else 0 for keyword in SQL_KEYWORDS])

    def __compute_similarities_torch(self, target_vector, ds_vectors):
        target_vector = torch.tensor(target_vector, dtype=torch.float32).to("cuda")
        ds_vectors = torch.tensor(ds_vectors, dtype=torch.float32).to("cuda")
        similarities = torch.nn.functional.cosine_similarity(target_vector, ds_vectors)
        return similarities.cpu().numpy()

    def select_examples(self, sample, dataset, n_examples):
        target_question, db_id = sample["question"], sample["db_id"]

        database_schema = DATABASE_CATALOG.get_database_schema_by_id(db_id)
        generated_query = self.__generate_sql_query(target_question, database_schema)

        target_vector = self.__encode_query_to_binary_vector(generated_query)

        precomputed_vectors = [
            self.__encode_query_to_binary_vector(sample["query"]) for sample in dataset
        ]
        precomputed_vectors = np.array(precomputed_vectors)

        cos_sims = self.__compute_similarities_torch(target_vector, precomputed_vectors)

        sorted_indices = np.argsort(-cos_sims)
        return [dataset[idx] for idx in sorted_indices[:n_examples]]
