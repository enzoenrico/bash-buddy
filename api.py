import http
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/v1")
async def not_main():
    return "not ok"

