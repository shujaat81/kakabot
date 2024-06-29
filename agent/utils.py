import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

from utils import define_device

load_dotenv(Path(".env"))


def get_embedding(text, embedding_model):
    """Get embeddings for a given text using the provided embedding model"""

    # Encode the text to obtain embeddings using the provided embedding model
    embedding = embedding_model.encode(text, show_progress_bar=False)

    # Convert the embeddings to a list of floats and return
    return embedding.tolist()


def map2embeddings(data, embedding_model):
    """Map a list of texts to their embeddings using the provided embedding model"""

    # Initialize an empty list to store embeddings
    embeddings = []

    # Iterate over each text in the input data list
    no_texts = len(data)
    print(f"Mapping {no_texts} pieces of information")
    for i in tqdm(range(no_texts)):
        # Get embeddings for the current text using the provided embedding model
        embeddings.append(get_embedding(data[i], embedding_model))

    # Return the list of embeddings
    return embeddings


def add_indefinite_article(role_name):
    """Check if a role name has a determinative adjective before it, and if not, add the correct one"""

    # Check if the first word is a determinative adjective
    determinative_adjectives = ["a", "an", "the"]
    words = role_name.split()
    if words[0].lower() not in determinative_adjectives:
        # Use "a" or "an" based on the first letter of the role name
        determinative_adjective = "an" if words[0][0].lower() in "aeiou" else "a"
        role_name = f"{determinative_adjective} {role_name}"

    return role_name


class LLMHF:
    """Wrapper for the Transformers implementation of Gemma"""

    def __init__(self, model_name, tokenizer_name, max_seq_length=2048):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_seq_length = max_seq_length

        # Initialize the model and tokenizer
        print("\nInitializing model:")
        self.device = define_device()
        self.model, self.tokenizer = self.initialize_model(
            self.model_name, self.device, self.max_seq_length
        )

    def initialize_model(self, model_name, device, max_seq_length):
        """Initialize a 4-bit quantized causal language model (LLM) and tokenizer with specified settings"""

        # Define the data type for computation
        compute_dtype = getattr(torch, "float16")

        hf_access_token = os.getenv("HF_TOKEN")

        # Define the configuration for quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

        # Load the pre-trained model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=hf_access_token,
            trust_remote_code=True,
            device_map=device,
        )

        # Load the tokenizer with specified device and max_seq_length
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            token=hf_access_token,
            device_map=device,
            trust_remote_code=True,
            max_seq_length=max_seq_length,
        )

        # Return the initialized model and tokenizer
        return model, tokenizer

    def generate_text(self, prompt, max_new_tokens=2048, temperature=0.0):
        """Generate text using the instantiated tokenizer and model with specified settings"""

        # Encode the prompt and convert to PyTorch tensor
        try:
            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
                self.device
            )

            # Determine if sampling should be performed based on temperature
            do_sample = True if temperature > 0 else False

            # Generate text based on the input prompt
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

            # Decode the generated output into text
            results = [self.tokenizer.decode(output) for output in outputs]
        except ValueError as v:
            print(f"Error during encoding the prompt:\n{v}")
            # Prepare the prompt
            tokenized_prompt = self.tokenizer(prompt)
            tokenized_prompt = torch.tensor(
                tokenized_prompt["input_ids"], device=self.device
            )

            tokenized_prompt = tokenized_prompt.unsqueeze(0)
            outputs = self.model.generate(
                tokenized_prompt,
                pad_token_id=0,
            )
        # Decode the generated output into text
        results = [self.tokenizer.decode(output) for output in outputs]
        # Return the list of generated text results
        return results
