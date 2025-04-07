import json
import torch

from typing import Optional
from src.inference.local_model import HuggingFaceModels
from src.utils.load_input import InputLoader
from src.utils.prompt_builder import Prompter


class CleanData:
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        prompt_template: Optional[dict] = None,
        huggingface_obj=HuggingFaceModels(
            model_name_or_path="TheBloke/Asclepius-13B-GPTQ",
            # model_name_or_path="starmpcc/Asclepius-13B",
        ),
    ):
        self.huggingface_obj = huggingface_obj
        self.huggingface_obj.device = device
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = InputLoader().load_file("prompt_tempaltes.yaml")
        pass

    def _build_prompt(
        self,
        query_text: str,
        prompt_template: dict,
    ):
        prompter = Prompter(prompt_template=prompt_template)
        return prompter.build_chat_prompt(query_text=query_text)

    def _abbreviation_expansion(
        self,
        text: str,
    ):
        text = f"[Discharge Summary Begin]\n{text}\n[Discharge Summary End]"
        prompt = self._build_prompt(
            query_text=text,
            prompt_template=self.prompt_template.get("abbreviation_expansion", ""),
        )
        abbr_response = self.huggingface_obj.generate(
            input_prompt_dict=prompt,
        )
        # TODO: Have retries if the json is not proper
        abbr_dict = json.loads(abbr_response)
        for key, value in abbr_dict.items():
            text.replace(old=key, new=value)
        return text

    def _normalize_drug_names(
        self,
        text: str,
    ):
        prompt = self._build_prompt(
            query_text=text,
            prompt_template=self.prompt_template.get("normalize_drug_names", ""),
        )
        normalized_drug_names_gen = self.huggingface_obj.generate(
            input_prompt_dict=prompt,
        )
        # TODO: Have retries if the json is not proper
        normalized_dict = json.loads(normalized_drug_names_gen)
        normalized_text = normalized_dict.get("normalized_text", "")
        return normalized_text

    def clean_data(
        self,
        data_point: list[str],
    ):
        # data_point = self._abbreviation_expansion(text=data_point)
        # data_point = self._normalize_drug_names(text=data_point)
        for line in data_point:
            line = self._abbreviation_expansion(text=line)
            # line = self._normalize_drug_names(text=line)

        data_point = "\n".join(data_point)
        return data_point
