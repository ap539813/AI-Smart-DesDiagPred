from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="KPI Analysis API", description="API for KPI Descriptive, Diagnostic, and Forecasting Analysis", version="1.0.0")

app.include_router(router)
