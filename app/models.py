from pydantic import BaseModel
from typing import List, Optional

class KPIAnalysisRequest(BaseModel):
    kpi_column: str = "RETAIL SALES"
    groupby_columns: Optional[List[str]] = None
    # forecast_periods: Optional[int] = 12

class KPIForecastingAnalysisRequest(BaseModel):
    kpi_column: str = "RETAIL SALES"
    forecast_periods: Optional[int] = 12

class KPIAnalysisResponse(BaseModel):
    results: str
    explanation: Optional[str]
