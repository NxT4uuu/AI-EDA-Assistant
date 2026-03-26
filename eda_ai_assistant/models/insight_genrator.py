from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch


class EDAInsightGenerator:

    def __init__(self):
        self.model_name = "t5-small"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def generate_insight(self, feature_text):

        prompt = f"Explain dataset feature: {feature_text}"

        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )

        output = self.model.generate(
            input_ids,
            max_length=60,
            num_beams=4,
            early_stopping=True
        )

        insight = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return insight