import json
import torch

from typing import Optional

from inference.local_model import HuggingFaceModels
from utils.load_input import InputLoader
from utils.prompt_builder import Prompter


class EntityExtractor:
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

    def _build_prompt(
        self,
        query_text: str,
        prompt_template: dict,
    ):
        prompter = Prompter(prompt_template=prompt_template)
        return prompter.build_chat_prompt(query_text=query_text)

    def _post_processor():
        pass

    def extract_entities(
        self,
        data_point: str,
    ):
        prompt = self._build_prompt(
            query_text=data_point,
            prompt_template=self.prompt_template.get("medical_entity_extraction", ""),
        )
        gen_response = self.huggingface_obj.generate(
            input_prompt_dict=prompt,
        )
        extracted_entities = json.loads(gen_response)

        # TODO: Post processor

        ######

        pass
