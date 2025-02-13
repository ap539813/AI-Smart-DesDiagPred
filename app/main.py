from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Retail Sales Analysis API", description="API for Sales (Descriptive, Diagnostic, and Forecasting) Analysis", version="1.0.0")

app.include_router(router)
