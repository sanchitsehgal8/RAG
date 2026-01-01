# this loader works page by page for pdf files  
# reads pdf files and converts them into document objects
# it reads as a list of document objects
# every document object has page content and metadata
#not great for scanned pdfs, better to use ocr loaders for that
# used when only text extraction from pdfs is needed
# use lazy loading to load multiple pdf files from a directory if its huge 

print("Langchain PDF Loader Example")
from langchain_community.document_loaders import PyPDFLoader
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
prompt = PromptTemplate(
    template="Write a short summary for the following poem:\n\n{poem}\n\nSummary:",
    input_variables=["poem"],
)
output_parser = StrOutputParser()

loader = PyPDFLoader("dl-curriculum.pdf")
documents = loader.load()

print(documents)
print(len(documents))


        
