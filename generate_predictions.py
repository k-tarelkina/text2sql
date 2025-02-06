from dataset import Dataset
from example_selection import MaskedQuestionSimilaritySelection, QuerySimilaritySelection, QuestionSimilaritySelection, RandomSelection
from few_shot_text2sql import FewShotText2SQL
from llm import LLM
from prompt_organization import DAILOrganization, FullInformationOrganization, SQLOnlyOrganization
from zero_shot_text2sql import ZeroShotText2SQL


def main():
    llm = LLM()

    dataset = Dataset()

    example_selections = {
        'Random': RandomSelection(llm),
        'QTS': QuestionSimilaritySelection(llm), 
        'MQS': MaskedQuestionSimilaritySelection(llm),
        'QRS': QuerySimilaritySelection(llm)
    }

    prompt_organizations = {
        'FI': FullInformationOrganization(),
        'SQL-only': SQLOnlyOrganization(),
        'DAIL': DAILOrganization()
    }

    n_examples = [2, 4, 8]

    configurations =  {
        'Zero-shot': ZeroShotText2SQL(llm)
    }

    for n in n_examples:
        for es_name, es in example_selections.items():
            for po_name, po in prompt_organizations.items():
                configurations[f'{es_name}({n}) + {po_name}'] = FewShotText2SQL(
                    llm, 
                    dataset=dataset, 
                    example_selection=es, 
                    prompt_organization=po, 
                    n_examples=n)

if __name__ == "__main__":
    main()