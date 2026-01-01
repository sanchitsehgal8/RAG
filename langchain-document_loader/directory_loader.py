# use lazy loading to load multiple pdf files from a directory if its huge 
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
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
loader = DirectoryLoader(
    path="books",
    glob= '**/*.pdf',
    loader_cls=PyPDFLoader

)
documents = loader.lazy_load()
print(documents)
print(len(documents))


# in load all docs are in the memory stored in lazy load docs are loaded one by one when iterated
#lazy laoding is useful when there are huge number of documents in a directory 