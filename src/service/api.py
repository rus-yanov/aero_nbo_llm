# src/service/api.py
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from src.service.nbo_pipeline import (
    get_nbo_response,
    get_nbo_response_from_rows,
)


class NboByClientRequest(BaseModel):
    client_id: int
    top_n: int = 3
    channel: str = "push"
    provider: Optional[str] = None


class NboByRowsRequest(BaseModel):
    rows: List[Dict[str, Any]]
    top_n: int = 3
    channel: str = "push"
    provider: Optional[str] = None
    client_id: Optional[int] = None


app = FastAPI(title="Aero NBO + LLM")


@app.post("/api/v1/nbo/by-client")
def nbo_by_client(req: NboByClientRequest):
    """
    Режим 1: всё берём из внутренних датасетов по client_id.
    """
    return get_nbo_response(
        client_id=req.client_id,
        top_n=req.top_n,
        channel=req.channel,
        provider=req.provider,
    )


@app.post("/api/v1/nbo/by-rows")
def nbo_by_rows(req: NboByRowsRequest):
    """
    Режим 2: внешняя система присылает нам готовые фичи (rows).
    """
    return get_nbo_response_from_rows(
        rows=req.rows,
        client_id=req.client_id,
        top_n=req.top_n,
        channel=req.channel,
        provider=req.provider,
    )