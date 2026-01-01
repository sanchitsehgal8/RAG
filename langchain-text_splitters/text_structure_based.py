from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """LangChain is a framework for developing applications powered by language models. It can be used for chatbots, Generative Question-Answering (GQA), summarization, and much more.
LangChain provides a standard interface for all LLMs, as well as a toolkit of components to build applications. It also includes integrations with other data sources and APIs to enhance the capabilities of language models."""

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = splitter.split_text(text)
print(texts)  # prints list of text chunks with a maximum of 100 characters each
print(len(texts))  # prints the number of text chunks generated
