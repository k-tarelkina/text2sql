from transformers import (
    BitsAndBytesConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
import torch
from dotenv import load_dotenv
import os

load_dotenv()


class LLM:
    def __init__(self, llm_name="mistralai/Ministral-8B-Instruct-2410"):
        hf_token = os.getenv("HF_TOKEN")

        self.__tokenizer = AutoTokenizer.from_pretrained(
            llm_name, padding_side="left", token=hf_token
        )
        self.__tokenizer.use_default_system_prompt = False
        self.__tokenizer.pad_token_id = self.__tokenizer.eos_token_id

        quantization_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=True)

        self.__llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=quantization_config,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
            token=os.getenv("HF_TOKEN"),
        )
        self.__llm.eval()

    def __parse_answer(self, answer: str) -> str:
        processed_answer = answer.strip()
        return processed_answer

    def answer(self, prompt: str, max_new_tokens=500) -> str:
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.__tokenizer.eos_token_id,
            pad_token_id=self.__tokenizer.pad_token_id,
        )

        turns = [{"role": "user", "content": prompt}]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = self.__tokenizer.apply_chat_template(turns, return_tensors="pt").to(
            device
        )

        with torch.no_grad():
            outputs = self.__llm.generate(input_ids, generation_config)

        answer_tokens = outputs[0, input_ids.shape[1] : -1]
        return self.__parse_answer(self.__tokenizer.decode(answer_tokens))

    def get_hidden_representation(self, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoding = self.__tokenizer.encode(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = self.__llm(encoding, output_hidden_states=True)

        last_layer_hidden_states = output.hidden_states[-2]
        last_token = last_layer_hidden_states[0, -1, :]

        return last_token.to("cpu").float().numpy()
