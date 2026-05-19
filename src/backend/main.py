from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.chat import router as chat_router
from routes.history import router as history_router
from routes.jobs import router as jobs_router
from routes.logs import router as logs_router
from services.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(title="Job Recommendation Bot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(logs_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
