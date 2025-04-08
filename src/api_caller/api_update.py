import requests
import os
import json


class Stadardize:
    def __init__(
        self,
        input_entities: dict,
    ):
        self.input_entities = input_entities

    def standardize_entities(
        self,
    ):
        standardized_entities = {}

        for key, entities in self.input_entities.items():

            standardized_entities[key] = [
                self._standardize_entity(
                    entity, "RXNORM" if key == "drugs" else "SNOMEDCT"
                )
                for entity in entities
            ]

    def _standardize_entity(
        self,
        entity: str,
        source: str,
    ):
        result = self._search(entity, source)
        results_list = result.get("result", {}).get("results", [])
        if not results_list:
            return entity
        best_match = results_list[0]
        standard_name = best_match.get("name", entity)
        return standard_name

    def _search(
        self,
        entity: str,
        source: str,
    ):
        ticket = self._get_ticket()
        service_ticket = self._get_service_ticket(ticket)
        url = "https://uts-ws.nlm.nih.gov/restsearch/current"
        params = {
            "string": entity,
            "ticket": service_ticket,
            "pageNumber": 1,
            "sabs": source,
            "returnIdType": "code",
        }
        response = requests.get(url, params=params)
        return response.json()
        pass

    def _get_service_ticket(
        self,
        ticket: str,
    ):
        params = {"service": "http://umlsks.nlm.nih.gov"}
        response = requests.post(ticket, data=params)
        return response.text

    def _get_ticket(
        self,
    ):
        params = {"apikey": os.getenv("UMLS_API_KEY")}
        response = requests.post(
            "https://utslogin.nlm.nih.gov/cas/v1/api-key", data=params
        )
        ticket = response.headers["location"]
        return ticket


if __name__ == "__main__":
    input_entities = {
        "drugs": ["Arthrotec"],
        "ades": [
            "bit drowsy",
            "little blurred vision",
            "gastric problems",
            "feel a bit weird",
        ],
        "symptoms_diseases": ["arthritis", "agony", "pains"],
    }
    standardize = Stadardize(input_entities)
    standardized_output = standardize.standardize_entities()
    print(json.dumps(standardized_output, indent=4))
    pass
