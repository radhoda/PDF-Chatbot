import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFium2Loader
from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


class PDFExplorer:
    def __init__(self, user_api_key=None):
        self.embeddings = OpenAIEmbeddings(openai_api_key=user_api_key)
        # environment variable for better security
        os.environ["OPENAI_API_KEY"] = user_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.llm = ChatOpenAI(temperature=0, openai_api_key=user_api_key)
        self.chain = None
        self.db = None
