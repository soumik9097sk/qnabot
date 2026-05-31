
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from vectordb.chroma_manager import get_vectorstore
from dotenv import load_dotenv
import os

load_dotenv()
YOUR_API_KEY = os.getenv("YOUR_API_KEY")

def upload_doc(data, document_type='md'):
    if document_type=='md':
        headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ]

        # Initialize splitter and process the downloaded file
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        split_chunks = markdown_splitter.split_text(data)

        vectordb = get_vectorstore()
        vectordb.add_documents(split_chunks)
