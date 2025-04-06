import torch

from typing import Optional

from inference.local_model import HuggingFaceModels
from utils.load_input import InputLoader
from utils.prompt_builder import Prompter
from utils.post_processor import PostProcessor


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
        self.error_log = []
        self.post_processor = PostProcessor()
        if self.prompt_template is None:
            self.prompt_template = InputLoader().load_file("prompt_tempaltes.yaml")

    def _build_prompt(
        self,
        query_text: str,
        prompt_template: dict,
    ):
        prompter = Prompter(prompt_template=prompt_template)
        return prompter.build_chat_prompt(query_text=query_text)

    def _ground_truth_extractor(
        self,
        data_point: str,
    ) -> dict[list]:
        ground_truth = {
            "drugs": [],
            "ades": [],
            "symptoms_diseases": [],
        }

        lines = data_point.split("\n")

        for line in lines:
            words = line.split()
            if words[1].lower() == "drug".lower():
                ground_truth["drugs"].append(words[4])
            elif words[1].lower() == "ADR".lower():
                ground_truth["ades"].append(words[4])
            else:
                ground_truth["symptoms_diseases"].append(words[4])

        return ground_truth

    def _post_processor(
        self,
        gen_response: str,
        ground_truth_dict: dict,
    ):
        parsable, error_log = self.post_processor.post_processor(
            response_str=gen_response,
            ground_truth_dict=ground_truth_dict,
        )
        return parsable, error_log
        pass

    def extract_entities(
        self,
        data_point: str,
    ):
        ground_truth_dict = self._ground_truth_extractor(data_point=data_point)
        prompt = self._build_prompt(
            query_text=data_point,
            prompt_template=self.prompt_template.get("medical_entity_extraction", ""),
        )
        gen_response = self.huggingface_obj.generate(
            input_prompt_dict=prompt,
        )

        # TODO: Post processor

        ######

        pass
