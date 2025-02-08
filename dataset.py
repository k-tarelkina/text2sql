from collections.abc import Sequence
from typing import Any, Dict, Literal
from datasets import load_dataset
from tqdm import tqdm
from llm import LLM
import nltk
from nltk import pos_tag

nltk.download("averaged_perceptron_tagger_eng")


class Dataset(Sequence):
    def __init__(
        self, llm: LLM, split: Literal["train", "validation"], limit=None
    ) -> None:
        self.ds = load_dataset("xlangai/spider")[split]

        if limit is not None:
            self.ds = self.ds.select(range(limit))

        self.ds = self.ds.to_list()

        for sample in tqdm(self.ds):
            sample["question_vector"] = llm.get_hidden_representation(
                sample["question"]
            )
            sample["masked_question_vector"] = llm.get_hidden_representation(
                self.__mask_question(sample)
            )

    def __mask_question(self, sample):
        question_toks = sample["question_toks"]
        pos_tags = pos_tag(question_toks)
        masked_question_toks = [
            "[MASK]" if pos.startswith("NN") or pos.startswith("JJ") else word
            for word, pos in pos_tags
        ]
        masked_question = " ".join(masked_question_toks)
        return masked_question

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.ds[index]

    def __contains__(self, item: Dict[str, Any]) -> bool:
        return item in self.ds

    def __iter__(self):
        return iter(self.ds)

    def __reversed__(self):
        return reversed(self.ds)
