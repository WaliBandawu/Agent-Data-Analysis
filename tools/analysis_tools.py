# tools/analysis_tools.py
import pandas as pd
from langchain.tools import tool
import os

DATA_PATH = "data/"

@tool("dataset_summary", return_direct=True)
def dataset_summary(file_name: str):
    """Generate comprehensive summary statistics for a dataset."""
    try:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            return {"error": "Dataset not found."}
        
        df = pd.read_csv(path)
        
        # Convert pandas dtypes to strings for JSON serialization
        data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        # Convert missing values to regular Python ints
        missing_values = {col: int(count) for col, count in df.isnull().sum().items()}
        
        # Basic info
        info = {
            "file_name": file_name,
            "shape": list(df.shape),  # Convert tuple to list
            "columns": list(df.columns),
            "data_types": data_types,
            "missing_values": missing_values,
            "memory_usage": int(df.memory_usage(deep=True).sum())  # Convert to regular int
        }
        
        # Statistical summary for numeric columns
        numeric_summary = {}
        if not df.select_dtypes(include=['number']).empty:
            desc = df.describe()
            numeric_summary = {
                col: {stat: float(value) for stat, value in desc[col].items()}
                for col in desc.columns
            }
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_summary = {}
        for col in categorical_cols:
            top_values = df[col].value_counts().head()
            categorical_summary[col] = {
                "unique_values": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top_values.items()}
            }
        
        info.update({
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary
        })
        
        return info
        
    except Exception as e:
        return {"error": f"Failed to generate summary: {str(e)}"}