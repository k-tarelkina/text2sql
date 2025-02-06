from abc import abstractmethod


class Text2SQL:
  '''
  Abstract Text2SQL model
  '''
  @abstractmethod
  def generate_sql(self, sample) -> str:
    pass