import os
import json
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv

from src.prompt_builder import Prompter

env_file = ".env.dev"
load_dotenv(env_file)
login(os.getenv("HUGGING_FACE_KEY"))


class HuggingFaceModels:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        pass

    def _load_model(
        self,
    ):
        try:
            model_config = {
                "pretrained_model_name_or_path": self.model_name_or_path,
                "device_map": "auto",
            }
            model = AutoModelForCausalLM.from_pretrained(**model_config)
        except:
            raise ValueError("Issue loading model. Contact the admin.")
        return model

    def _load_tokenizer(
        self,
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        except:
            raise ValueError("Issue loading tokenizer. Contact the admin.")
        return tokenizer

    def _build_prompt(
        self,
        query_text: str,
        prompt_template: dict,
    ):
        prompter = Prompter(prompt_template=prompt_template)
        return prompter.build_chat_prompt(query_text=query_text)

    def generate(
        self,
        query_text: str,
        prompt_template: dict[str, any],
    ) -> dict:
        model = self._load_model()
        tokenizer = self._load_tokenizer()
        input_prompt_dict = self._build_prompt(
            query_text=query_text, prompt_template=prompt_template
        )
        input_text = tokenizer.apply_chat_template(
            input_prompt_dict, tokenize=False, add_generation_prompt=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = model.generate(**inputs, max_new_tokens=20)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded_output
