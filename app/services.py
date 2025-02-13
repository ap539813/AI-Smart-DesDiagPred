import pandas as pd
import numpy as np
import shap
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import json
import ollama
from app.utils import LLAMA_MODEL, DEEPSEEK_MODEL, QWEN_MODEL

model_name = DEEPSEEK_MODEL

class DescriptiveAnalysisService:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    
    def create_date_column(self, df):
        df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
        return df
    
    def preprocess_data(self, data_path, kpi_column, groupby_columns=None):
        df = pd.read_csv(data_path)
        
        if groupby_columns:
            group_columns = ["YEAR", "MONTH"] + groupby_columns
        else:
            group_columns = ["YEAR", "MONTH"]
        
        df = df.groupby(group_columns).agg({kpi_column: "sum"}).reset_index()
        df = self.create_date_column(df)
        df.sort_values("DATE", inplace=True)
        return df
    
    def descriptive_analysis(self, df, kpi_column, groupby_columns=None):
        threshold = self.config["kpi_threshold"]
        ratechange_threshold = self.config["ratechange_threshold"]
        
        df["Above_Threshold"] = df[kpi_column] > threshold
        df["Rate_Change"] = df[kpi_column].pct_change()
        df["Rate_Above_Threshold"] = df["Rate_Change"] > ratechange_threshold
        
        latest_above_threshold = df[df["Above_Threshold"]].sort_values("DATE", ascending=False).iloc[0]
        
        summary = {
            "latest_period_above_threshold": {
                "month": latest_above_threshold["MONTH"],
                "year": latest_above_threshold["YEAR"],
                "value": latest_above_threshold[kpi_column]
            },
            "current_vs_last_year": df[kpi_column].pct_change(12).iloc[-1],
            "current_vs_last_month": df[kpi_column].pct_change().iloc[-1],
        }

        print(df.head())
        
        if groupby_columns:
            top_values = df[df["Above_Threshold"]][["MONTH"] + groupby_columns + [kpi_column] + ["DATE"]]
            top_values_sorted = top_values.sort_values(by=["DATE", kpi_column], ascending=[False, False]).head(10)
            summary["top_10_values_above_threshold"] = top_values_sorted.to_dict(orient="records")
            
            top_rate_changes = df[df["Rate_Above_Threshold"]][["MONTH"] + groupby_columns + ["Rate_Change"] + ["DATE"]]
            top_rate_changes_sorted = top_rate_changes.sort_values(by=["DATE", "Rate_Change"], ascending=[False, False]).head(10)
            summary["top_10_ratechange_above_threshold"] = top_rate_changes_sorted.to_dict(orient="records")
            
            min_row = df.loc[df[kpi_column].idxmin()]
            max_row = df.loc[df[kpi_column].idxmax()]
            summary["min_value"] = {"value": min_row[kpi_column], "month": min_row["MONTH"], "group": {col: min_row[col] for col in groupby_columns}}
            summary["max_value"] = {"value": max_row[kpi_column], "month": max_row["MONTH"], "group": {col: max_row[col] for col in groupby_columns}}
        else:
            top_values = df[df["Above_Threshold"]][["MONTH"] + [kpi_column] + ["DATE"]]
            top_values_sorted = top_values.sort_values(by=["DATE", kpi_column], ascending=[False, False]).head(10)
            summary["top_10_values_above_threshold"] = top_values_sorted.to_dict(orient="records")
            
            top_rate_changes = df[df["Rate_Above_Threshold"]][["MONTH"] + ["Rate_Change"] + ["DATE"]]
            top_rate_changes_sorted = top_rate_changes.sort_values(by=["DATE", "Rate_Change"], ascending=[False, False]).head(10)
            summary["top_10_ratechange_above_threshold"] = top_rate_changes_sorted.to_dict(orient="records")
            summary["min_value"] = df[kpi_column].min()
            summary["max_value"] = df[kpi_column].max()
        
        return str(summary)
    
    def generate_explanation(self, result):
        prompt = f"Analyze the following KPI results and provide a business explanation with inferences: {result}\n"
        prompt += """
        
        INSTRUCTIONS:
        - Do not think 
        - Use the provided data only
        - Do the month_name-year conversion for the date related data present in the results
        - Do not use any other data
        - Write the output exactly as shown below

        OUTPUT FORMAT:
        <output>
            <overview>overview of the lates period above threshold</overview>
            <periods>Overview of the periods exceeding the threshold</periods>
            <ratechange-analysis>Rate change analysis</ratechange-analysis>
            <year-on-year>Year on year analysis</year-on-year>
            <min-max>Min-max analysis</min-max>
        </output>
        
        """
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]



class DiagnosticAnalyticsService:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    
    def preprocess_data(self, data_path, groupby_columns=None):
        df = pd.read_csv(data_path)
        
        if groupby_columns:
            group_columns = groupby_columns
            df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
            df.sort_values("DATE", inplace=True)
            df = df.groupby(group_columns)
            return df
        else:
            df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
            df.sort_values("DATE", inplace=True)
            df.dropna(inplace=True)
            return df
    
    def detect_anomaly(self, df, kpi_column):
        threshold = self.config["kpi_threshold"]
        anomalies = df.groupby(["YEAR", "MONTH"])[kpi_column].sum().reset_index()
        anomalies = anomalies[anomalies[kpi_column] > threshold]
        return anomalies[["YEAR", "MONTH"]].values.tolist()
    
    def reverse_label_encoding(self, value, le):
        return le.inverse_transform([value])[0] if le else value
    
    def diagnostic_analysis(self, df, kpi_column, feature_columns, groupby_columns=None):
        anomalies = self.detect_anomaly(df, kpi_column)
        if not anomalies:
            return "No anomalies detected."
        
        latest_anomaly = anomalies[-1]
        year, month = latest_anomaly
        df_subset = df[(df["YEAR"] == year) & (df["MONTH"] == month)]
        
        label_encoders = {}
        for col in feature_columns:
            le = LabelEncoder()
            df_subset[col] = le.fit_transform(df_subset[col])
            label_encoders[col] = le
        
        X = df_subset[feature_columns]
        y = df_subset[kpi_column]
        
        model = RandomForestRegressor()
        model.fit(X, y)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # plt.figure(figsize=(10, 5))
        # shap.summary_plot(shap_values, X)
        
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        top_features = np.argsort(feature_importance)[-3:][::-1]
        
        results = {"Anomaly Period": {"Year": year, "Month": month}}
        
        for i, feature_idx in enumerate(top_features):
            feature_name = feature_columns[feature_idx]
            top_values = df_subset[feature_name].value_counts().head(3).index.tolist()
            top_kpi_values = {self.reverse_label_encoding(val, label_encoders.get(feature_name)): df_subset[df_subset[feature_name] == val][kpi_column].sum() for val in top_values}
            results[f"Top Contributing Factor {i+1}"] = {"Feature": feature_name, "Values": top_kpi_values}
            subset_df = df_subset[df_subset[feature_name] == top_values[0]]
            X_subset = subset_df[feature_columns]
            y_subset = subset_df[kpi_column]
            
            model_subset = RandomForestRegressor()
            model_subset.fit(X_subset, y_subset)
            explainer_subset = shap.Explainer(model_subset, X_subset)
            shap_values_subset = explainer_subset(X_subset)
            
            feature_importance_subset = np.abs(shap_values_subset.values).sum(axis=0)
            top_sub_feature_idx = np.argsort(feature_importance_subset)[-1]
            top_sub_feature = feature_columns[top_sub_feature_idx]
            
            top_sub_values = subset_df[top_sub_feature].value_counts().head(1).index.tolist()
            top_sub_kpi_value = subset_df[subset_df[top_sub_feature] == top_sub_values[0]][kpi_column].sum()
            
            results[f"Top Contributing Factor {i+1}"][f"Drill Down Top Feature"] = {
                "Feature": top_sub_feature, 
                "Value": self.reverse_label_encoding(top_sub_values[0], label_encoders.get(top_sub_feature)), 
                "Average KPI": top_sub_kpi_value
            }
        return results
    
    def generate_explanation(self, result):
        prompt = f"Analyze the following KPI diagnostic results and provide a business explanation with inferences: {result}"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]


class KPIForecastingService:
    def __init__(self, config_path):
        self.config = self.load_config(config_path)
    
    def load_config(self, config_path):
        with open(config_path, "r") as file:
            return json.load(file)
    
    def preprocess_data(self, data_path, kpi_column):
        df = pd.read_csv(data_path)
        df["DATE"] = pd.to_datetime(df["YEAR"].astype(str) + "-" + df["MONTH"].astype(str) + "-01") + pd.offsets.MonthEnd(0)
        df = df[["DATE", kpi_column]].groupby("DATE").sum().reset_index()
        df.sort_values("DATE", inplace=True)
        return df
    
    def forecast_kpi(self, df, kpi_column, forecast_periods=12, confidence_interval=0.95):
        df.set_index("DATE", inplace=True)
        
        model = SARIMAX(df[kpi_column], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit()
        
        forecast_results = model_fit.get_forecast(steps=forecast_periods)
        forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=forecast_periods, freq="M")
        forecast_mean = forecast_results.predicted_mean
        confidence_intervals = forecast_results.conf_int(alpha=1 - confidence_interval)
        
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[kpi_column], label="Historical KPI", color="blue")
        plt.plot(forecast_index, forecast_mean, label="Forecasted KPI", color="red", linestyle="dashed")
        plt.fill_between(forecast_index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color="pink", alpha=0.3, label=f"{int(confidence_interval*100)}% Confidence Interval")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("KPI Value")
        plt.title("KPI Forecast with Confidence Interval")
        plt.show()
        
        forecast_data = {
            "DATE": forecast_index.strftime('%Y-%m-%d').tolist(),
            "Forecast": forecast_mean.tolist(),
            "Lower Bound": confidence_intervals.iloc[:, 0].tolist(),
            "Upper Bound": confidence_intervals.iloc[:, 1].tolist()
        }
        
        explanation = self.generate_explanation(forecast_data)
        return {"results": forecast_data, "explanation": explanation}
    
    def generate_explanation(self, result):
        prompt = f"Analyze the following KPI forecast results and provide a business explanation with inferences: {result}"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]