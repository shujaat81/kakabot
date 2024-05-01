import pandas as pd

from agent.agent_classes import AIAssistant
from agent.utils import GemmaHF
from embedding_pipeline.wiki_utils import get_wikipedia_pages

categories = [
    "Machine_learning",
    "Data_science",
    "Statistics",
    "Deep_learning",
    "Artificial_intelligence",
]
extracted_texts = get_wikipedia_pages(categories)
print("Found", len(extracted_texts), "Wikipedia pages")

wikipedia_data_science_kb = pd.DataFrame(extracted_texts, columns=["wikipedia_text"])
wikipedia_data_science_kb.to_csv("wikipedia_data_science_kb.csv", index=False)
wikipedia_data_science_kb.head()

# Initialize the name of the embeddings and model
embeddings_name = "thenlper/gte-large"
model_name = "/kaggle/input/gemma/transformers/2b-it/1"

# Create an instance of AIAssistant with specified parameters
gemma_ai_assistant = AIAssistant(
    gemma_model=GemmaHF(model_name), embeddings_name=embeddings_name
)

# Map the intended knowledge base to embeddings and index it
gemma_ai_assistant.learn_knowledge_base(knowledge_base=extracted_texts)

# Save the embeddings to disk (for later use)
gemma_ai_assistant.save_embeddings()

# Set the temperature (creativity) of the AI assistant and set the role
gemma_ai_assistant.set_temperature(0.0)
gemma_ai_assistant.set_role(
    "data science expert whose explanations are useful, clear and complete"
)

gemma_ai_assistant.query(
    "What is the difference between data science, machine learning, and artificial intelligence?"
)

gemma_ai_assistant.query("Explain how linear regression works")

gemma_ai_assistant.query(
    "What are decision trees, and how do they work in machine learning?"
)

gemma_ai_assistant.query(
    "What is cross-validation, and why is it used in machine learning?"
)

gemma_ai_assistant.query(
    "Explain the concept of regularization and its importance in preventing overfitting in machine learning models"
)
