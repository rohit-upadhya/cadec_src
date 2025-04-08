import json

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.data_types import ErrorTypes


class PostProcessor:
    def __init__(
        self,
        error_log: list = [],
    ):
        self.error_log = error_log

    def _syntactic_validator(
        self,
        response_str: str,
    ):
        final_dict = {}
        parsable = True
        try:
            final_dict = json.loads(response_str)
        except Exception as e:
            self._log_errors(
                error_type=ErrorTypes.JSON_PARSING_ERROR,
                message=f"{e}",
            )
            parsable = False
        return final_dict, parsable

    def _completeness_validor(
        self,
        final_gen_dict: dict,
        ground_truth_dict: dict,
    ):
        for key, ground_values in ground_truth_dict.items():
            final_gen_list = final_gen_dict.get(key, [])
            for item in ground_values:
                if item not in final_gen_list:
                    self._log_errors(
                        error_type=(
                            ErrorTypes.DRUG_MISSING
                            if key.lower() == "drugs"
                            else (
                                ErrorTypes.SYMPTOM_MISSING
                                if key.lower() == "drugs"
                                else ErrorTypes.ADE_MISSING
                            )
                        ),
                        message=item,
                    )
                    pass

        pass

    def _semantic_validator(
        self,
        final_dict: dict,
        data_point: str | list,
    ):
        all_words = []
        for k, v in final_dict.items():
            all_words.append(v)
        all_words = " ".join(all_words)
        original = ""
        if isinstance(data_point, list):
            original = " ".join(data_point)
        else:
            original = data_point
        sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = sentence_transformer_model.encode([original, all_words])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        if similarity < 0.5:
            self._log_errors(
                error_type=ErrorTypes.SEMANTIC_MISMATCH,
                message=f"{original}",
                detail=f"{all_words}",
            )
        del sentence_transformer_model
        pass

    def post_processor(
        self,
        response_str: str,
        ground_truth_dict: dict,
        data_point: str,
    ):
        final_dict, self.parsable = self._syntactic_validator(
            response_str=response_str,
        )
        if self.parsable:
            self._completeness_validor(
                final_gen_dict=final_dict,
                ground_truth_dict=ground_truth_dict,
            )
            self._semantic_validator(
                final_dict=final_dict,
                data_point=data_point,
            )
        print("Post processor completed with the following errors : ", self.error_log)
        return self.parsable, self.error_log, final_dict
        pass

    def _log_errors(
        self,
        error_type: ErrorTypes,
        message: str = "",
        detail: str = "",
    ):
        self.error_log.append(
            {
                "error_type": error_type.value,
                "message": message,
                "detail": detail,
            }
        )
        pass
