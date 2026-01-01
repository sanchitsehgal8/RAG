# reads text files and converts them into document objects
# it reads as a list of document objects
# every document object has page content and metadata
# use lazy loading to load multiple pdf files from a directory if its huge 
print("Langchain Text Loader Example")
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import sys
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ['HF_HOME'] = 'D:/huggingface_cache'
print("[DEBUG] Script started", flush=True)
time.sleep(0.1)


print("[DEBUG] Loading model...", flush=True)
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
print("[DEBUG] Model loaded", flush=True)
model = ChatHuggingFace(llm=llm)
prompt = PromptTemplate(
    template="Write a short summary for the following poem:\n\n{poem}\n\nSummary:",
    input_variables=["poem"],
)
output_parser = StrOutputParser()

print("[DEBUG] Loading cricket.txt...", flush=True)
loader = TextLoader("cricket.txt")
documents = loader.load()
print("[DEBUG] Document loaded", flush=True)
print(documents[0], flush=True)
print(type(documents), flush=True)
print(len(documents), flush=True)
print(type(documents[0]), flush=True)
print(documents[0].page_content, flush=True)
print(documents[0].metadata, flush=True)

print("[DEBUG] Running chain...", flush=True)
chain = prompt|model|output_parser
summary = chain.invoke({"poem": documents[0].page_content})
print("Summary:", flush=True)
print(summary, flush=True)