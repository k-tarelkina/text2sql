from abc import abstractmethod
from typing import List


class Text2SQL:
  @abstractmethod
  def generate_sql(self, question: str, question_tokens: List[str], db_id: str) -> str:
    pass