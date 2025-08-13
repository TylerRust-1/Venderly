# fill_missing.py
from llama_cpp import Llama

class GapFiller:
    def __init__(self, model_path: str):
        self.llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)

    def fill(self, record: dict, mandatory_fields: list):
        """
        Fill missing (non-mandatory) fields in a dictionary using the LLaMA model.
        """
        for field, value in record.items():
            if (field not in mandatory_fields) and (value is None or value == ""):
                prompt = (
                    f"You are filling missing survey form data.\n"
                    f"Known data: { {k:v for k,v in record.items() if v not in [None,'']} }\n"
                    f"Field to fill: {field}\n"
                    f"Provide a plausible value. Keep it short."
                )
                output = self.llm(prompt, max_tokens=30, stop=["\n"])
                text = output["choices"][0]["text"].strip()
                record[field] = text
        return record
