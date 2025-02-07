from database_catalog import DATABASE_CATALOG
from llm import LLM
from text2sql import Text2SQL


class ZeroShotText2SQL(Text2SQL):
    def __init__(self, llm: LLM):
        self.__llm = llm

        self.prompt_template = """
      You are a very competent SQL agent.
      ### Complete sqlite SQL query only and with no explanation in only one line.
      ### Avoid using JOIN and its alternatives except when there is no other possibility.
      ### Do not use "as".
      ### Go for the simplest solution
      ### Database schema:
      {schema}
      ### {question} SELECT
    """.strip()

    def generate_sql(self, sample) -> str:
        prompt = self.prompt_template.format(
            schema=DATABASE_CATALOG.get_database_schema_by_id(sample["db_id"]),
            question=sample["question"],
        )

        return self.__llm.answer(prompt)
