import json


import re

def read_json(json_data: str | dict):
    """Prosta funkcja do odczytu JSON cytologii (obsługuje też ```json ... ```)."""
    if isinstance(json_data, dict):
        return json_data
    if isinstance(json_data, str):
        s = json_data.strip()
        # usuń znaczniki ```json oraz ```
        s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*```$', '', s)
        return json.loads(s)
    raise TypeError(f"Unsupported type: {type(json_data)}")

json_data = "```json\n{\n  \"cells\": [\n    {\n      \"id\": \"0\",\n      \"predicted_class\": \"NSIL\",\n      \"explanation\": \"Characterized by a very low nucleus-to-cell area ratio (NCr=0.0011) and a small nuclear size (N=154), consistent with a benign squamous cell.\",\n      \"confidence\": 1.0,\n      \"needs_expert_review\": false\n    },\n    {\n      \"id\": \"1\",\n      \"predicted_class\": \"NSIL\",\n      \"explanation\": \"Exhibits a low nucleus-to-cell area ratio (NCr=0.0167) and nuclear features within benign limits (N=3524, EqN=66.98).\",\n      \"confidence\": 1.0,\n      \"needs_expert_review\": false\n    },\n    {\n      \"id\": \"2\",\n      \"predicted_class\": \"NSIL\",\n      \"explanation\": \"Demonstrates a low nucleus-to-cell area ratio (NCr=0.0175) and nuclear dimensions consistent with benign cytology (N=3132, EqN=63.15).\",\n      \"confidence\": 1.0,\n      \"needs_expert_review\": false\n    },\n    {\n      \"id\": \"3\",\n      \"predicted_class\": \"NSIL\",\n      \"explanation\": \"Shows a low nucleus-to-cell area ratio (NCr=0.0271) and nuclear features typical of a benign squamous cell (N=2807, EqN=59.78).\",\n      \"confidence\": 0.98,\n      \"needs_expert_review\": false\n    },\n    {\n      \"id\": \"4\",\n      \"predicted_class\": \"NSIL\",\n      \"explanation\": \"Presents a low nucleus-to-cell area ratio (NCr=0.0304) and benign nuclear morphology (N=3656, EqN=68.23).\",\n      \"confidence\": 0.99,\n      \"needs_expert_review\": false\n    }\n  ],\n  \"slide_summary\": {\n    \"overall_class\": \"NSIL\",\n    \"explanation\": \"The slide contains five squamous cells, all classified as Negative for Squamous Intraepithelial Lesion (NSIL). All cells exhibit low nucleus-to-cell area ratios (NCr ranging from 0.0011 to 0.0304) and nuclear sizes within benign parameters, lacking features of atypia, koilocytosis, or significant nuclear enlargement associated with LSIL or HSIL.\",\n    \"confidence\": 0.98\n  }\n}\n```"

data = read_json(json_data)
print(f"Klasa: {data['slide_summary']['overall_class']}")