from dotenv import load_dotenv

load_dotenv()
from gpt4all import GPT4All

# Initialize the model
model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

# Prompt for summarization with examples
prompt = f"""
Describe the following movie using emojis.

Titanic: 🛳️🌊❤️🧊🎶🔥🚢💔👫💑
The Matrix: 🕶️💊💥👾🔮🌃👨🏻‍💻🔁🔓💪

Toy Story: """

# Generate the response
response = model.generate(
    prompt=prompt, max_tokens=60
)  # Adjust max_tokens if necessary

print(response)
