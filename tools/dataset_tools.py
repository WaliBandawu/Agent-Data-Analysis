# tools/dataset_tools.py
import os
import pandas as pd
from langchain.tools import tool
import math
import numpy as np

DATA_PATH = "data/"

@tool("list_datasets", return_direct=True)
def list_datasets(input: str):
    """List all available CSV datasets in the /data folder."""
    try:
        os.makedirs(DATA_PATH, exist_ok=True)
        files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]
        return {"datasets": files, "count": len(files)}
    except Exception as e:
        return {"error": f"Failed to list datasets: {str(e)}"}

@tool("load_dataset", return_direct=True)
def load_dataset(file_name: str):
    """Load and preview dataset"""
    try:
        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            return {"error": f"Dataset {file_name} not found."}

        df = pd.read_csv(path)
        
        # Convert preview data to JSON-safe format
        preview_data = df.head().to_dict('records')
        
        # Clean NaN values in preview data
        def clean_preview_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_preview_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_preview_nan(item) for item in obj]
            elif pd.isna(obj):  # Handle pandas NaN
                return None
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            else:
                return obj
        
        clean_preview = clean_preview_nan(preview_data)
        
        return {
            "file_name": file_name,
            "shape": list(df.shape),
            "columns": list(df.columns),
            "preview": clean_preview,
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

    except Exception as e:
        return {"error": str(e)}