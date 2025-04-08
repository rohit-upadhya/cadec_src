import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from dotenv import load_dotenv

env_file = ".env.dev"
load_dotenv(env_file)
# login(os.getenv("HUGGING_FACE_KEY"))


class HuggingFaceModels:
    def __init__(
        self,
        model_name_or_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        pass

    def _load_model(
        self,
    ):
        try:
            model_config = {
                "pretrained_model_name_or_path": self.model_name_or_path,
                "device_map": self.device,
            }
            quantization_config = self._quantization_config()
            if "meta-llama" not in self.model_name_or_path:
                model_config["quantization_config"] = quantization_config
            model = AutoModelForCausalLM.from_pretrained(**model_config)
        except:
            raise ValueError("Issue loading model. Contact the admin.")
        return model

    def _quantization_config(
        self,
    ):
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # )
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        return bnb_config

    def _load_tokenizer(
        self,
    ):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        except:
            raise ValueError("Issue loading tokenizer. Contact the admin.")
        return tokenizer

    #     {'role': 'system', 'content': 'You are a very smart medical entity extractor. Your job is to extract all the : \n  - Drugs (Medication Names)\n  - Adverse Drug Events (ADEs)\n  - Symptoms/Diseases\nfrom the provided text.\nYou will extract all of them, and will not leave out any at all.\nYour output should have\n{\n  "drugs": ["<drug 1>","drug 2",...],\n  "ades": ["ade1","ade2",...],\n  "symptoms_diseases": ["<symptoms_disease_1>","<symptoms_disease_2>",...]\n}\n## Note : if there are no entries for any of the 3 entities, return an empty list for them.\n'}
    # {'role': 'user', 'content': 'T1 Symptom 153 157 pain\nT2 ADR 0 9 heartburn\nT3 ADR 11 17 nausea\nT4 ADR 40 56 voracious hunger\nT5 ADR 59 103 sharp unbearable cramping pains in lower gut\nT6 Symptom 234 253 pain in my forearms'}
    def _apply_chat_template(
        self,
        input_prompt_dict_list: list[dict],
    ):
        prompt_text = ""

        for prompt_item in input_prompt_dict_list:
            if "system" in prompt_item["role"]:
                prompt_text = f"{prompt_text}{prompt_item['content']}\n"
            elif "user" in prompt_item["role"]:
                prompt_text = f"{prompt_text}\n{prompt_item['content']}\n"
            elif "assistant" in prompt_item["role"]:
                prompt_text = f"Your previous response was : {prompt_text}\n{prompt_item['content']}\n"
        # prompt_text = f"{prompt_text}<|start_header_id|>assistant<|end_header_id|>"
        return prompt_text

    def _apply_chat_template_llama(
        self,
        input_prompt_dict_list: list[dict],
    ):
        prompt_text = ""

        for prompt_item in input_prompt_dict_list:
            if "system" in prompt_item["role"]:
                prompt_text = f"{prompt_text}<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{prompt_item['content']}<|eot|>\n"
            elif "user" in prompt_item["role"]:
                prompt_text = f"{prompt_text}<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt_item['content']}<|eot|>\n"
            elif "assistant" in prompt_item["role"]:
                prompt_text = f"{prompt_text}<|start_header_id|>assistant<|end_header_id|>\n{prompt_item['content']}<|eot|>\n"
        prompt_text = f"{prompt_text}<|start_header_id|>assistant<|end_header_id|>"
        return prompt_text.strip()
        pass

    def generate(
        self,
        input_prompt_dict: list[dict],
    ) -> dict:
        chat_template_to_apply = (
            self._apply_chat_template
            if "meta-llama" in self.model_name_or_path
            else self._apply_chat_template
        )
        input_text = chat_template_to_apply(input_prompt_dict)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=300)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoded_output = decoded_output.split(input_text)
        decoded_output = (
            decoded_output[0] if len(decoded_output) == 1 else decoded_output[1]
        )
        decoded_output = decoded_output.split("<|eot|>")[0]
        return decoded_output
