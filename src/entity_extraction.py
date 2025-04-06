import torch
import json

from typing import Optional
from copy import deepcopy

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

    def _get_prompter(
        self,
        prompt_template: dict,
    ):
        return Prompter(prompt_template=prompt_template)
        return

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
        parsable, error_log, final_dict = self.post_processor.post_processor(
            response_str=gen_response,
            ground_truth_dict=ground_truth_dict,
        )
        return parsable, error_log, final_dict
        pass

    def extract_entities(
        self,
        data_point: str,
    ):
        ground_truth_dict = self._ground_truth_extractor(data_point=data_point)
        prompter = self._get_prompter(
            prompt_template=self.prompt_template.get("medical_entity_extraction", "")
        )
        prompt = prompter.build_chat_prompt(query_text=data_point)
        gen_response = self.huggingface_obj.generate(
            input_prompt_dict=prompt,
        )
        prompt.append(
            {
                "role": "assistant",
                "content": gen_response,
            }
        )
        # TODO: Post processor
        parsable, error_log, final_dict = self._post_processor(
            gen_response=gen_response, ground_truth_dict=ground_truth_dict
        )
        log = {
            "data_point": data_point,
            "successful": False,
            "ground_truth": ground_truth_dict,
            "error_log": [error_log],
            "model_responses": [gen_response],
            "prompt_history": prompt,
            "prompter": prompter,
            "last_parsed": None,
        }
        if parsable:
            log["last_parsed"] = final_dict

        regen_no = 0
        while len(log["error_log"][-1]) > 0 and regen_no < 3:
            log = self._rereun_extract(log=deepcopy(log))
            regen_no += 1

        self._logger(
            log=log,
        )
        return log["last_parsed"]

    def _rereun_extract(
        self,
        log: dict[str, any],
    ):
        updated_prompt = log["prompter"].reprompter(
            current_prompt=log["prompt_history"],
            error_log=log["error_log"],
        )
        gen_response = self.huggingface_obj.generate(
            input_prompt_dict=updated_prompt,
        )
        updated_prompt.append(
            {
                "role": "assistant",
                "content": gen_response,
            }
        )
        log["prompt_history"] = updated_prompt
        parsable, error_log, final_dict = self._post_processor(
            gen_response=gen_response, ground_truth_dict=log["ground_truth"]
        )
        if parsable:
            log["last_parsed"] = final_dict
        log["error_log"].append(error_log)
        return deepcopy(log)

    def _logger(
        self,
        log: dict[str, any],
    ):
        log["final_output"] = {}
        if log.get("error_log", [])[-1] == 0:
            log["successful"] = True
            log["final_output"] = log["last_parsed"]
        log.pop("prompter", None)
        json.dump(
            log,
            sort_keys=False,
            ensure_ascii=False,
            indent=4,
        )
