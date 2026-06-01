from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
from src.utils.loader import load_pdf

from src.services.qa import qa

app = FastAPI(title="Simple QA Backend")


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str


@app.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    try:
        answer = qa(question)
        return QAResponse(answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def save_pdf(file: UploadFile) -> str:
    pdf_path = UPLOAD_DIR / file.filename

    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # load pdf
    load_pdf(pdf_path)

    return str(pdf_path)


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    pdf_path = save_pdf(file)

    return {"message": "PDF uploaded successfully", "path": pdf_path}



from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")