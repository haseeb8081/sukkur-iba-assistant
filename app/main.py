from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.core.config import settings

app = FastAPI(title="Sukkur IBA University Chatbot")

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}", "status": "online"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Routes
from app.api.endpoints import router as api_router
app.include_router(api_router, prefix="/api", tags=["rag"])

# Serve Frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
