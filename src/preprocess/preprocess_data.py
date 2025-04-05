import os


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

    def preprocess_data(
        self,
    ):
        data_files = self._load_data_files(data_dir=self.data_dir)
        data_points = self._load_datapoints(data_files=data_files)
        return data_points
