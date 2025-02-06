from llm import LLM
from text2sql import Text2SQL


class ZeroShotText2SQL(Text2SQL):
  def __init__(self, llm: LLM):
    self.llm = llm
  
  def generate_sql(self, question: str, question_tokens: str[], db_id: str):
    return ""
