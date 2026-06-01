from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
YOUR_API_KEY = os.getenv("YOUR_API_KEY")


def upload_doc(data, document_type="md"):
    if not YOUR_API_KEY:
        raise RuntimeError("Missing YOUR_API_KEY environment variable.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=YOUR_API_KEY)
    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    if document_type == "md":
        if isinstance(data, (list, tuple)):
            # If the loader already returned Document objects, add them directly.
            if data and hasattr(data[0], "page_content"):
                vector_store.add_documents(data)
                return
            text_data = [str(item) for item in data]
            vector_store.add_texts(text_data)
            return

        if isinstance(data, str):
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            markdown_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            split_chunks = markdown_splitter.split_text(data)
            vector_store.add_texts(split_chunks)
            return

    raise TypeError(
        "Unsupported data type for upload_doc. Expected str or list of documents."
    )
