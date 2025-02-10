from abc import abstractmethod

from src.datasets.database_catalog import DATABASE_CATALOG


class PromptOrganization:
    @abstractmethod
    def get_prompt(self, sample, examples) -> str:
        pass


class FullInformationOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
            Given the following database schema:
            {database_schema}

            Answer the following: {question}
            {query}
            """.strip()

    def get_prompt(self, sample, examples) -> str:
        result_prompt = """
        Provide sqlite SQL query only and with no explanation.
        Avoid using JOIN and its alternatives except when there is no other possibility.
        Do not use "as".
        Do not use aliases for table names if possible.
        Go for the simplest solution.
        """

        for example in examples:
            result_prompt += self.prompt_template.format(
                question=example["question"],
                query=example["query"],
                database_schema=DATABASE_CATALOG.get_database_schema_by_id(
                    example["db_id"]
                ),
            )
            result_prompt += "\n"

        result_prompt += self.prompt_template.format(
            question=sample["question"],
            query="",
            database_schema=DATABASE_CATALOG.get_database_schema_by_id(sample["db_id"]),
        )

        return result_prompt.strip()


class SQLOnlyOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
        Given the following database schema:
        {database_schema}

        Answer the following: {question}

        Some SQL examples are provided based on similar problems:
        {example_queries}

        Provide sqlite SQL query only and with no explanation.
        Avoid using JOIN and its alternatives except when there is no other possibility.
        Do not use "as".
        Do not use aliases for table names if possible.
        Go for the simplest solution.
    """.strip()

    def get_prompt(self, sample, examples) -> str:
        example_queries = ""

        for example in examples:
            example_queries += example["query"]
            example_queries += "\n"

        result_prompt = self.prompt_template.format(
            example_queries=example_queries,
            question=sample["question"],
            database_schema=DATABASE_CATALOG.get_database_schema_by_id(sample["db_id"]),
        )

        return result_prompt.strip()


class DAILOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
        Given the following database schema:
        {database_schema}

        Answer the following: {question}

        Some example questions and corresponding SQL queries are provided based on similar problems : */
        {examples}

        Provide sqlite SQL query only and with no explanation.
        Avoid using JOIN and its alternatives except when there is no other possibility.
        Do not use "as".
        Do not use aliases for table names if possible.
        Go for the simplest solution.
        """.strip()

    def get_prompt(self, sample, examples) -> str:
        example_queries = ""

        example_prompt_template = """
            Question : {question}
            Answer : {query}
            """.strip()

        for example in examples:
            example_queries += example_prompt_template.format(
                question=example["question"], query=example["query"]
            )
            example_queries += "\n"

        result_prompt = self.prompt_template.format(
            examples=example_queries,
            question=sample["question"],
            database_schema=DATABASE_CATALOG.get_database_schema_by_id(sample["db_id"]),
        )

        return result_prompt
