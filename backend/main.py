
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from api.v1.routers import search

app = FastAPI(
    title="LangGraph API",
    description="LangGraph API for building and deploying stateful applications",
    version="1.0.0",
)

app.include_router(search.router, prefix="/api/v1/search", tags=["search"])