import os
import json
import torch

from typing import Optional

from src.load_input import InputLoader
from src.local_model import HuggingFaceModels


class PreprocessData:
    def __init__(
        self,
        prompt_template: Optional[dict] = None,
        data_dir: str = "dataset/Original",
        device: torch.device = torch.device("cpu"),
    ):
        self.data_dir = data_dir
        self.input_loader = InputLoader()
        self.prompt_template = prompt_template
        if self.prompt_template is None:
            self.prompt_template = self.input_loader.load_file("prompt_tempaltes.yaml")
        self.huggingface_obj = HuggingFaceModels(
            model_name_or_path="TheBloke/Asclepius-13B-GPTQ",
            # model_name_or_path="starmpcc/Asclepius-13B",
            prompt_template=self.prompt_template,
            device=device,
        )

    def _load_data_files(
        self,
        data_dir: str,
    ):
        files = []
        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if "pdf" in filename:
                    files.append(os.path.join(dirpath, filename))
        return files

    def _load_datapoints(
        self,
        data_files: str,
    ) -> list[list[str]]:
        extracted_data = []

        for file in data_files:
            with open(file, "r") as stream:
                data_point = stream.read()
                data_point = self._extract_relevant_section(data_point)
                extracted_data.append(data_point)
        return extracted_data

    def _extract_relevant_section(
        self,
        data_point: str,
    ):
        lines = data_point.split("\n")
        updated_lines = []
        for item in lines:
            if item.startswith("#"):
                continue
            updated_lines.append(item)
        return updated_lines

    def _abbreviation_expansion(
        self,
        text: str,
    ):
        gen_text = self.huggingface_obj.generate(
            query_text=text,
            prompt_template=self.prompt_template.get("abbreviation_expansion", ""),
        )
        # TODO: Have retries if the json is not proper
        abbr_dict = json.loads(gen_text)
        for key, value in abbr_dict.items():
            text.replace(old=key, new=value)
        return gen_text

    def _normalize_drug_names(self, text: str):
        normalized_drug_names_gen = self.huggingface_obj.generate(
            query_text=text,
            prompt_template=self.prompt_template.get("normalize_drug_names", ""),
        )
        # TODO: Have retries if the json is not proper
        normalized_dict = json.loads(normalized_drug_names_gen)
        return normalized_dict.get("normalized_text", "")

    def preprocess_data(
        self,
    ):
        data_files = self._load_data_files(data_dir=self.data_dir)
        data_points = self._load_datapoints(data_files=data_files)

        for point in data_points:
            for line in point:
                line = self._abbreviation_expansion(text=line)
                line = self._normalize_drug_names(text=line)
            point = point.join("\n")
        return data_points
