from abc import abstractmethod


class Text2SQL:
  @abstractmethod
  def generate_sql(self, sample) -> str:
    pass