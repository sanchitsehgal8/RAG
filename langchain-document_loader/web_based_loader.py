# Web based document loader
# reads web pages and converts them into document objects
# it reads as a list of document objects
# every document object has page content and metadata
#uses beautifulsoup4 and requests libraries to fetch and parse web pages    
# if js heavy use selenium or playwright based loader instead

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
import sys
import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
os.environ['HF_HOME'] = 'D:/huggingface_cache'
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)
loader= WebBaseLoader("https://en.wikipedia.org/wiki/Brigitte_Bardot")
documents = loader.load()
print(documents)    
print(len(documents))