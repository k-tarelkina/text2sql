from src.datasets.database_catalog import DATABASE_CATALOG
from src.llm import LLM
from src.strategy.text2sql import Text2SQL
from src.utils.log import Logger


class ZeroShotText2SQL(Text2SQL):
    def __init__(self, llm: LLM, logger: Logger):
        self.__llm = llm
        self.__logger = logger

        self.prompt_template = """
            You are a very competent SQL agent.
            Provide sqlite SQL query only and with no explanation.
            Avoid using JOIN and its alternatives except when there is no other possibility.
            Do not use "as".
            Do not use aliases for table names if possible.
            Go for the simplest solution.

            Database schema:
            {schema}

            Question:
            {question}
    """.strip()

    def generate_sql(self, sample) -> str:
        self.__logger.write("Start generating prompt")
        prompt = self.prompt_template.format(
            schema=DATABASE_CATALOG.get_database_schema_by_id(sample["db_id"]),
            question=sample["question"],
        )
        self.__logger.write("End generating prompt")

        self.__logger.write("Start asking LLM")
        result = self.__llm.answer(prompt)
        self.__logger.write("End asking LLM")

        return result
