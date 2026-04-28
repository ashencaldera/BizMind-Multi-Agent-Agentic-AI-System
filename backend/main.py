"""
BizMind — Autonomous Multi-Agent Business Intelligence & Decision System
FastAPI Backend Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.routes import router
from core.config import config

app = FastAPI(
    title="BizMind API",
    description="Autonomous Multi-Agent Business Intelligence & Decision System",
    version=config.VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend on any port during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes under /api/v1
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "app": "BizMind",
        "version": config.VERSION,
        "status": "running",
        "docs": "/docs",
        "api": "/api/v1",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc), "type": type(exc).__name__},
    )


if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════╗
║        BizMind v{config.VERSION} — Starting Up         ║
║  Autonomous Business Intelligence AI    ║
╚══════════════════════════════════════════╝
    """)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.DEBUG,
        log_level="info",
    )
