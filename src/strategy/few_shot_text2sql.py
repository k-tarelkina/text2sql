from src.datasets.dataset import Dataset
from src.llm import LLM
from src.strategy.prompt_organization import PromptOrganization
from src.strategy.text2sql import Text2SQL
from src.strategy.example_selection import ExampleSelection
from src.utils.log import Logger


class FewShotText2SQL(Text2SQL):
    def __init__(
        self,
        llm: LLM,
        dataset: Dataset,
        example_selection: ExampleSelection,
        prompt_organization: PromptOrganization,
        logger: Logger,
        n_examples=5,
    ):

        self.__llm = llm
        self.__example_selection = example_selection
        self.__prompt_organization = prompt_organization
        self.__n_examples = n_examples
        self.__dataset = dataset
        self.__logger = logger

    def generate_sql(self, sample) -> str:
        self.__logger.write("Start generating prompt")
        examples = self.__example_selection.select_examples(
            sample, self.__dataset, self.__n_examples
        )
        prompt = self.__prompt_organization.get_prompt(sample, examples)
        self.__logger.write(f"End generating prompt, prompt: {prompt}")

        self.__logger.write("Start asking LLM")
        result = self.__llm.answer(prompt)
        self.__logger.write(f"End asking LLM, answer: {result}")

        return result
