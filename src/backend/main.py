from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.chat import router as chat_router
from routes.jobs import router as jobs_router

app = FastAPI(title="Job Recommendation Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
