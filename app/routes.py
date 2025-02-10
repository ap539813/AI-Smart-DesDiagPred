from fastapi import APIRouter
from app.services import DiagnosticAnalyticsService, KPIForecastingService, forecasting_analysis
from app.models import KPIAnalysisRequest, KPIAnalysisResponse, KPIForecastingAnalysisRequest
from app.services import DescriptiveAnalysisService

router = APIRouter()
kpi_service = DescriptiveAnalysisService(config_path="config.json")
kpi_diagnostic_service = DiagnosticAnalyticsService(config_path="config.json")
kpi_forecasting_service = KPIForecastingService(config_path="config.json")


@router.post("/descriptive", response_model=KPIAnalysisResponse)
async def generate_kpi_explanation(request: KPIAnalysisRequest):
    df = kpi_service.preprocess_data("Warehouse_and_Retail_Sales_enhanced.csv", request.kpi_column, groupby_columns=request.groupby_columns)
    result = kpi_service.descriptive_analysis(df, request.kpi_column, groupby_columns=request.groupby_columns)
    explanation = kpi_service.generate_explanation(result)
    return KPIAnalysisResponse(results=result, explanation=explanation)



@router.post("/diagnostic", response_model=KPIAnalysisResponse)
async def run_diagnostic_analysis(request: KPIAnalysisRequest):
    if request.groupby_columns:
        feature_columns=[col for col in ["SUPPLIER", "REGION", "MANAGER"] if col not in request.groupby_columns]
    else:
        feature_columns=["SUPPLIER", "REGION", "MANAGER"]
    if not request.groupby_columns:
        df = kpi_diagnostic_service.preprocess_data("Warehouse_and_Retail_Sales_enhanced.csv")
        results = kpi_diagnostic_service.diagnostic_analysis(df, request.kpi_column, feature_columns=["SUPPLIER", "REGION", "MANAGER"])
        explanation = kpi_diagnostic_service.generate_explanation(results)
    else:
        df_groups = kpi_diagnostic_service.preprocess_data("Warehouse_and_Retail_Sales_enhanced.csv", groupby_columns=request.groupby_columns)
        results = {}
        explanation = ''
        for group, df_group in df_groups:
            df_group.dropna(inplace=True)
            result = kpi_diagnostic_service.diagnostic_analysis(df_group, request.kpi_column, feature_columns=feature_columns, groupby_columns=request.groupby_columns)
            results[group] = result
            explanation = kpi_diagnostic_service.generate_explanation(results) + '\n\n'
    return KPIAnalysisResponse(results=str(results), explanation=explanation)


@router.post("/forecast")
async def run_forecasting_analysis(request: KPIForecastingAnalysisRequest):
    df = kpi_forecasting_service.preprocess_data("Warehouse_and_Retail_Sales_enhanced.csv", request.kpi_column)
    response = kpi_forecasting_service.forecast_kpi(df, request.kpi_column, forecast_periods=request.forecast_periods)
    return response
