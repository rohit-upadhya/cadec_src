import torch
import gc

from src.preprocess.preprocess_data import PreprocessData
from src.preprocess.clean_data import CleanData
from src.entity_extraction import EntityExtractor
from src.inference.local_model import HuggingFaceModels
from src.api_update import Stadardize


class ADEExtractor:
    def __init__(
        self,
        data_dir: str = "dataset/test",
        device: str = "cuda",
    ):
        preprocess_data = PreprocessData(data_dir=data_dir)
        self.data_points = preprocess_data.preprocess_data()
        self.device: torch.device = torch.device(device)
        pass

    def _clean_data(
        self,
        hugging_face_object,
        data_point,
    ):

        data_cleaner = CleanData(
            device=self.device, huggingface_obj=hugging_face_object
        )
        cleaned_data_point = data_cleaner.clean_data(data_point)
        return cleaned_data_point

    def _make_processed_and_cleaned_dataset(
        self,
    ):
        processed_and_cleaned_data_points = []
        huggingface_obj = HuggingFaceModels(
            device=self.device,
            model_name_or_path="starmpcc/Asclepius-13B",
        )
        for data_point in self.data_points:
            cleaned_data = self._clean_data(
                hugging_face_object=huggingface_obj, data_point=data_point
            )
            processed_and_cleaned_data_points.append(
                {
                    "original": data_point,
                    "cleaned": cleaned_data,
                }
            )
        self._delete_object(obj=huggingface_obj)

        return processed_and_cleaned_data_points

    def _entity_extraction(
        self,
        processed_and_cleaned_data_points,
    ):
        huggingface_obj = HuggingFaceModels(
            device=self.device,
            model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        )
        entity_extractor = EntityExtractor(
            huggingface_obj=huggingface_obj,
            device=self.device,
        )

        extracted_data_points = []
        for data_point in processed_and_cleaned_data_points:
            extracted_datapoint = entity_extractor.extract_entities(
                data_point=data_point
            )
            updated_data_point = {k: v for k, v in data_point}
            updated_data_point["extracted_datapoint"] = extracted_datapoint
            extracted_data_points.append(updated_data_point)
        self._delete_object(obj=huggingface_obj)
        return extracted_data_points

    def _update_using_api(
        self,
        extracted_data_points,
    ):
        standardized_output_data_points = []
        for data_point in extracted_data_points:
            standardize = Stadardize(data_point["extracted_datapoint"])
            standardized_output = standardize.standardize_entities()
            updated_data_point = {k: v for k, v in data_point}
            updated_data_point["standardized_output"] = standardized_output
            standardized_output_data_points.append(updated_data_point)
        return standardized_output_data_points

    def ade_extraction(
        self,
    ):
        processed_and_cleaned_data_points = self._make_processed_and_cleaned_dataset()
        extracted_data_points = self._entity_extraction(
            processed_and_cleaned_data_points=processed_and_cleaned_data_points
        )
        api_updated_data_points = self._update_using_api(
            extracted_data_points=extracted_data_points
        )
        return api_updated_data_points

    def _delete_object(self, obj: object):
        del obj.model
        del obj
        gc.collect()

        torch.cuda.empty_cache()
