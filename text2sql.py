from abc import abstractmethod


class Text2SQL:
  @abstractmethod
  def generate_sql(self, question: str, question_tokens: str[], db_id: str):
    pass