# TODO: following steps need to be enabled for the pipeline
# Aggregate source documents
# Clean the document content
# Load document contents into memory. Tools like Unstructured, LlamaIndex, and LangChain's Document loaders
# Split the content into chunks
# Create embeddings for text chunks
# Store embeddings in a vector store

import os
from getpass import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions

# load pdf file
loader = PyPDFLoader("/Users/hasan/llm_projects/kakabot/knowledge_base/T_1.pdf")

pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(pages[10:20])

# get your free access token from HuggingFace and paste it here
HF_token = getpass()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_token

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token, model_name="BAAI/bge-base-en-v1.5"
)

client = weaviate.Client(embedded_options=EmbeddedOptions())

vectorstore = Weaviate.from_documents(
    client=client, documents=chunks, embedding=embedding_model, by_text=False
)

print("done")

retriever = vectorstore.as_retriever()

from langchain.prompts import ChatPromptTemplate

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)
