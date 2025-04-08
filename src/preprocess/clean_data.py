import json
import torch

from typing import Optional
from src.inference.local_model import HuggingFaceModels
from src.utils.load_input import InputLoader
from src.utils.prompt_builder import Prompter


class CleanData:
    def __init__(
        self,
        huggingface_obj: HuggingFaceModels,
        device: torch.device = torch.device("cpu"),
        prompt_template: Optional[dict] = None,
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
        print("abbr_response", abbr_response)
        # TODO: Have retries if the json is not proper

        # for key, value in abbr_dict.items():
        #     text.replace(old=key, new=value)
        return abbr_response

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
        print("normalized_drug_names_gen", normalized_drug_names_gen)
        # TODO: Have retries if the json is not proper
        # normalized_dict = json.loads(normalized_drug_names_gen)
        # normalized_text = normalized_dict.get("normalized_text", "")
        return normalized_drug_names_gen

    def clean_data(
        self,
        data_point: list[str],
    ):
        data_point = " ".join(data_point)
        abbr_data_point = self._abbreviation_expansion(text=data_point)
        norm_data_point = self._normalize_drug_names(text=f"Datapoint : {data_point}\nAbbreviation Expanded : {abbr_data_point}")
        new_data_point = f"{abbr_data_point}\n{norm_data_point}"
        # new_data_point = []
        # for line in data_point:
        #     new_line = self._abbreviation_expansion(text=line)
        #     drug_norm = ""
        #     if "drug" in line.lower():
        #         drug_norm = self._normalize_drug_names(text=line)
        #     updated_string = f"{new_line}\n{drug_norm}"
        #     updated_string = updated_string.strip()
        #     new_data_point.append(updated_string)

        # new_data_point = "\n".join(new_data_point)
        return new_data_point
