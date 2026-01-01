from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("langchain-text_splitters/dl-curriculum.pdf")
docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0,separator ='')
texts = splitter.split_documents(docs)
print(texts[0].page_content)  # prints list of Document objects with text chunks of 100 characters each 