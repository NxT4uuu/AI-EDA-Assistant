import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import streamlit as st


# Cache model (prevents reload + improves performance)
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Force CPU (stable deployment)
    model = model.to("cpu")
    
    return tokenizer, model


class EDAInsightGenerator:

    def __init__(self):
        self.tokenizer, self.model = load_model()

    def generate_insight(self, feature_text):

        # Safety check (avoid crashes on bad input)
        if not feature_text or len(feature_text.strip()) == 0:
            return "No meaningful information available."

        prompt = f"Explain dataset feature: {feature_text}"

        input_ids = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        )

        # Optimized generation (faster + less CPU)
        output = self.model.generate(
            input_ids,
            max_length=60,
            num_beams=2,  
            early_stopping=True
        )

        insight = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return insight