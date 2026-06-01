from langchain_community.document_loaders import PyPDFLoader


def load_pdf(document_path):
    loader = PyPDFLoader(document_path)
    docs = loader.load()

    try:
        from src.services.ingestion import upload_doc
    except Exception as exc:
        raise RuntimeError(
            "Unable to import upload function for PDF ingestion. "
            "Check src/services/ingestion.py and its dependencies."
        ) from exc

    upload_doc(docs, document_type="md")
