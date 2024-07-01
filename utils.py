import re

import numpy as np
import torch

from agent_classes.utils import add_indefinite_article
from embedding_pipeline.utils import get_embedding

BRACES_PATTERN = re.compile(r"\{.*?\}|\}")


def generate_summary_and_answer(
    question,
    data,
    searcher,
    embedding_model,
    model,
    max_new_tokens=2048,
    temperature=0.4,
    role="expert",
):
    """Generate an answer for a given question using context from a dataset"""

    # Embed the input question using the provided embedding model
    embeded_question = np.array(get_embedding(question, embedding_model)).reshape(1, -1)

    # Find similar contexts in the dataset based on the embedded question
    neighbors, distances = searcher.search_batched(embeded_question)

    # Extract context from the dataset based on the indices of similar contexts
    context = " ".join([data[pos] for pos in np.ravel(neighbors)])

    # Get the end-of-sentence token from the tokenizer
    try:
        EOS_TOKEN = model.tokenizer.eos_token
    except:
        EOS_TOKEN = "<eos>"

    # Add a determinative adjective to the role
    role = add_indefinite_article(role)

    # Generate a prompt for summarizing the context
    prompt = (
        f"""
             Summarize this context: "{context}" in order to answer the question "{question}" as {role}\
             SUMMARY:
             """.strip()
        + EOS_TOKEN
    )

    # Generate a summary based on the prompt
    results = model.generate_text(prompt, max_new_tokens, temperature)

    # Clean the generated summary
    summary = clean_text(results[0].split("SUMMARY:")[-1], EOS_TOKEN)

    # Generate a prompt for providing an answer
    prompt = (
        f"""
             Here is the context: {summary}
             Using the relevant information from the context 
             and integrating it with your knowledge,
             provide an answer as {role} to the question: {question}.
             If the context doesn't provide
             any relevant information answer with 
             [I couldn't find a good match in my
             knowledge base for your question, 
             hence I answer based on my own knowledge] \
             ANSWER:
             """.strip()
        + EOS_TOKEN
    )

    # Generate an answer based on the prompt
    results = model.generate_text(prompt, max_new_tokens, temperature)

    # Clean the generated answer
    answer = clean_text(results[0].split("ANSWER:")[-1], EOS_TOKEN)

    # Return the cleaned answer
    return answer


def save_embeddings_to_disk(embeddings, filename="/knowledge_base/embeddings.npy"):
    """Save the embeddings to disk"""
    np.save(filename, embeddings)


def define_device():
    """Define the device to be used by PyTorch"""

    # Get the PyTorch version
    torch_version = torch.__version__

    # Print the PyTorch version
    print(f"PyTorch version: {torch_version}", end=" -- ")

    # Check if MPS (Multi-Process Service) device is available on MacOS
    if torch.backends.mps.is_available():
        # If MPS is available, print a message indicating its usage
        print("using MPS device on MacOS")
        # Define the device as MPS
        defined_device = torch.device("mps")
    else:
        # If MPS is not available, determine the device based on GPU availability
        defined_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Print a message indicating the selected device
        print(f"using {defined_device}")

    # Return the defined device
    return defined_device


def clean_text(txt, EOS_TOKEN):
    """Clean text by removing specific tokens and redundant spaces"""
    txt = (
        txt.replace(
            EOS_TOKEN, ""
        )  # Replace the end-of-sentence token with an empty string
        .replace("**", "")  # Replace double asterisks with an empty string
        .replace("<pad>", "")  # Replace "<pad>" with an empty string
        .replace("  ", " ")  # Replace double spaces with single spaces
    ).strip()  # Strip leading and trailing spaces from the text
    return txt


def remove_braces_and_content(text):
    """Remove all occurrences of curly braces and their content from the given text"""
    return BRACES_PATTERN.sub("", text)


def clean_string(input_string):
    """Clean the input string."""

    # Remove extra spaces by splitting the string by spaces and joining back together
    cleaned_string = " ".join(input_string.split())

    # Remove consecutive carriage return characters until there are no more consecutive occurrences
    cleaned_string = re.sub(r"\r+", "\r", cleaned_string)

    # Remove all occurrences of curly braces and their content from the cleaned string
    cleaned_string = remove_braces_and_content(cleaned_string)

    # Return the cleaned string
    return cleaned_string
