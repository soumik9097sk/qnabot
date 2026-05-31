from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()
YOUR_API_KEY = os.getenv("YOUR_API_KEY")


def qa(user_question: str) -> str:
    if not user_question or not user_question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not YOUR_API_KEY:
        raise RuntimeError("Missing YOUR_API_KEY environment variable.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=YOUR_API_KEY)

    vector_store = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=YOUR_API_KEY, temperature=0)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know and tell a joke instead.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    response = rag_chain.invoke({"input": user_question})
    if isinstance(response, dict):
        return response.get("answer") or response.get("output") or str(response)
    return str(response)
