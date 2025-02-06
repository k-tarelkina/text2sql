from abc import abstractmethod

from database_catalog import DATABASE_CATALOG


class PromptOrganization:
    @abstractmethod
    def get_prompt(self, sample, examples) -> str:
        pass

class FullInformationOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
            /* Given the following database schema : */
            {database_schema}
            /* Answer the following : {question} */
            {query}
            """.strip()
        
    def get_prompt(self, sample, examples) -> str:
        result_prompt = ""

        for example in examples:
            result_prompt += self.prompt_template.format(
                question=example['question'],
                query=example['query'],
                database_schema=DATABASE_CATALOG.get_database_schema_by_id(example['db_id'])
            )
            result_prompt += "\n"

        result_prompt += self.prompt_template.format(
                question=sample['question'],
                query="SELECT",
                database_schema=DATABASE_CATALOG.get_database_schema_by_id(sample['db_id'])
            )

        result_prompt += "\n/* Complete SQL query only and with no explanation */"
        return result_prompt.strip()
    

class SQLOnlyOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
        /* Some SQL examples are provided based on similar problems : */
        {example_queries}
        /* Question : */
        {question}
        SELECT
        """.strip()
        
    def get_prompt(self, sample, examples) -> str:
        example_queries = ""

        for example in examples:
            example_queries += example['query']
            example_queries += "\n"

        result_prompt = self.prompt_template.format(example_queries=example_queries, question=sample['question'])
        result_prompt += "\n/* Complete SQL query only and with no explanation */"

        return result_prompt.strip()
    

class DAILOrganization(PromptOrganization):
    def __init__(self):
        self.prompt_template = """
            /* Answer the following : {question} */
            {query}
            """.strip()
        
    def get_prompt(self, sample, examples) -> str:
        example_queries = ""

        for example in examples:
            example_queries += example['query']
            example_queries += "\n"

        result_prompt = "/* Some example questions and corresponding SQL queries are provided based on similar problems : */\n"
        result_prompt = self.prompt_template.format(question=sample['question'], query=sample['query'])
        result_prompt += "\n/* Complete SQL query only and with no explanation */"

        return result_prompt