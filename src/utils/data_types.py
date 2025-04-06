from enum import Enum


class ErrorTypes(Enum):
    JSON_PARSING_ERROR = "json_parsing_error"
    DRUG_MISSING = "drug_missing_completeness_error"
    SYMPTOM_MISSING = "symptom_missing_completeness_error"
    ADE_MISSING = "ade_missing_completeness_error"
