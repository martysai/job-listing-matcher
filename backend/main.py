import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from routes.admin import router as admin_router
from routes.auth import COOKIE_NAME, router as auth_router, verify_cookie
from routes.chat import router as chat_router
from routes.history import router as history_router
from routes.logs import router as logs_router
from services.database import init_db
from services.log_sink import JsonlHandler
from services.recommender import get_recommender
from services.vacancy_refresh import VacancyRefreshService

_PUBLIC_PATHS = {"/api/auth/login", "/api/auth/logout", "/health"}

_sara_handler = JsonlHandler()
logging.getLogger("sara").addHandler(_sara_handler)
logging.getLogger("sara").setLevel(logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Build the recommender eagerly so any configuration error (missing
    # VACANCIES_PATH, broken Chroma dir, etc.) surfaces at startup instead
    # of on the first user request.  The same singleton is then re-used by
    # the conversation streamer and the background refresh loop.
    recommender = get_recommender()

    # Schedule the Adzuna refresh loop.  start() is no-throw: it silently
    # disables itself if credentials are missing or VACANCY_REFRESH_ENABLED
    # is false, so a misconfigured deployment still boots and serves from
    # the existing index.
    refresh_service = VacancyRefreshService(recommender)
    await refresh_service.start()
    app.state.vacancy_refresh = refresh_service

    try:
        yield
    finally:
        await refresh_service.stop()


app = FastAPI(title="Job Recommendation Bot API", lifespan=lifespan)


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS or not path.startswith("/api/"):
            return await call_next(request)
        if not verify_cookie(request.cookies.get(COOKIE_NAME, "")):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(_AuthMiddleware)

app.include_router(auth_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(history_router, prefix="/api")
app.include_router(logs_router, prefix="/api")
app.include_router(admin_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
