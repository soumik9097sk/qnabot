from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
