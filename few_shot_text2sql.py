from database_catalog import DATABASE_CATALOG
from dataset import Dataset
from example_selection import ExampleSelection
from llm import LLM
from prompt_organization import PromptOrganization
from text2sql import Text2SQL


class FewShotText2SQL(Text2SQL):
  def __init__(self, 
               llm: LLM, 
               dataset:Dataset,
               example_selection: ExampleSelection, 
               prompt_organization: PromptOrganization,
               n_examples=5):
    
    self.__llm = llm
    self.__example_selection = example_selection
    self.__prompt_organization = prompt_organization
    self.__n_examples = n_examples
    self.__dataset = dataset
  
  def generate_sql(self, sample) -> str:
    examples = self.__example_selection.select_examples(sample, self.__dataset, self.__n_examples)

    prompt = self.__prompt_organization.get_prompt(sample, examples)
    
    return self.__llm.answer(prompt)
