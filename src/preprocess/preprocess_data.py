import os
import re


class PreprocessData:
    def __init__(
        self,
        data_dir: str = "dataset/Original",
    ):
        self.data_dir = data_dir

    def _load_data_files(
        self,
        data_dir: str,
    ):
        files = []
        for dirpath, _, filenames in os.walk(data_dir):
            for filename in filenames:
                if ".ann" in filename:
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
        postition_in_text_pattern = re.compile(r"^\d+(;\d+)*$")
        for line in lines:
            line = line.replace("\t", " ")
            if line.startswith("#") or line == "":
                continue
            words = line.split()
            filtered_words = [
                t for t in words if not postition_in_text_pattern.match(t)
            ]
            line = " ".join(filtered_words[1:])
            updated_lines.append(line)
        return updated_lines

    def preprocess_data(
        self,
    ):
        data_files = self._load_data_files(data_dir=self.data_dir)
        data_points = self._load_datapoints(data_files=data_files)
        return data_points


if __name__ == "__main__":
    preprocess_data = PreprocessData(data_dir="dataset/test")
    data_points = preprocess_data.preprocess_data()
    print(data_points)
