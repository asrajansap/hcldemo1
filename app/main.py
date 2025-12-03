# app/main.py
import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.schemas import DumpIn, AnalysisOut
from app.llm_client import LLMClient, LLMError
from app.storage import Storage
from app.prompts import PROMPT_TEMPLATE, JSON_RESPONSE_EXAMPLE

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("st22-gateway")

app = FastAPI(title="SAP ST22 AI Gateway", version="1.0")

# Allow CORS for local UI during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize components
STORAGE_DB = os.environ.get("STORAGE_DB", "data/st22.db")
store = Storage(STORAGE_DB)

llm_client = LLMClient()

# -----------------------------------------------------------
# HEALTH CHECK (AI Core will call this first)
# -----------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


# -----------------------------------------------------------
# REQUIRED BY SAP AI CORE: /v1/predict
# -----------------------------------------------------------
@app.post("/v1/predict")
async def predict(request: Request):
    """
    Mandatory endpoint for SAP AI Core ServingTemplate.
    Accepts: { "prompt": "<string>" }
    Returns: { "result": { ... LLM result ... } }
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="'prompt' field is required")

    logger.info("AI Core /v1/predict called")

    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None, llm_client.analyze, prompt
        )
        return {"result": resp}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------
# Your existing ST22 dump analysis endpoint
# -----------------------------------------------------------
@app.post("/api/dumps", response_model=AnalysisOut)
async def analyze_dump(payload: DumpIn, request: Request):
    """
    Accepts an ST22-style JSON payload (DumpIn), builds a prompt, calls the LLM, and saves the AI analysis.
    """
    dump_header = payload.dump_header or {}
    dump_id = dump_header.get("id")
    if not dump_id:
        raise HTTPException(status_code=400, detail="dump_header.id is required")

    # Build the prompt
    prompt = PROMPT_TEMPLATE.format(dump=payload.json(), code=payload.dump_code or "")

    try:
        logger.info("Calling LLM for dump_id=%s provider=%s", dump_id, llm_client.provider)
        llm_resp = await asyncio.get_event_loop().run_in_executor(
            None, llm_client.analyze, prompt
        )
    except LLMError as e:
        logger.exception("LLM failed")
        raise HTTPException(status_code=500, detail=f"LLM provider error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected LLM error")
        raise HTTPException(status_code=500, detail=str(e))

    ai_parsed = llm_resp.get("json") or llm_resp.get("parsed") or None
    if not ai_parsed:
        ai_parsed = {"raw_text": llm_resp.get("text")}

    # persist
    rec = await asyncio.get_event_loop().run_in_executor(
        None, store.save_analysis, dump_id, payload.dict(), llm_resp
    )

    out = {
        "dump_id": dump_id,
        "dump": payload.dict(),
        "ai_summary": ai_parsed,
        "priority": ai_parsed.get("priority") if isinstance(ai_parsed, dict) else "Unknown",
        "raw_llm": llm_resp,
    }
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=out)


@app.get("/api/dumps/{dump_id}", response_model=AnalysisOut)
async def get_dump(dump_id: str):
    rec = await asyncio.get_event_loop().run_in_executor(None, store.get_analysis, dump_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    return rec


@app.get("/api/dumps", response_model=list[AnalysisOut])
async def list_recent(limit: int = 50):
    recs = await asyncio.get_event_loop().run_in_executor(None, store.list_recent, limit)
    return recs
