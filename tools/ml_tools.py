# tools/ml_tools.py
import os
import json
import pandas as pd
import math
import numpy as np
from langchain.tools import tool
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from zoofs import GeneticOptimization
from autogluon.tabular import TabularPredictor

DATA_PATH = "data/"

##################################
# 1. CLEANING TOOL
##################################
@tool("clean_data", return_direct=True)
def clean_data(params: str):
    """
    Cleans dataset: handles missing values, encodes categoricals, and drops ID-like columns.
    
    params: {"file_name": "train.csv", "target": "Target"}
    Returns cleaned CSV path.
    """
    try:
        if isinstance(params, str):
            params = json.loads(params)
        
        file_name = params.get("file_name")
        target = params.get("target")

        if not file_name or not target:
            return {"error": "file_name and target required."}

        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            return {"error": "Dataset not found."}

        df = pd.read_csv(path)

        # Drop ID-like columns
        drop_cols = [c for c in ["ID", "user_id", "prediction_time"] if c in df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")

        # Handle missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("MISSING")
            else:
                df[col] = df[col].fillna(df[col].median())

        # Save cleaned data
        clean_file = "cleaned_" + file_name
        df.to_csv(os.path.join(DATA_PATH, clean_file), index=False)

        return {"message": "Data cleaned successfully", "file_name": clean_file, "shape": df.shape}

    except Exception as e:
        return {"error": str(e)}


##################################
# 2. FEATURE SELECTION TOOL
##################################
def objective_function_topass(model, X_train, y_train, X_valid, y_valid):      
    model.fit(X_train, y_train)  
    if hasattr(model, "predict_proba"):
        preds = model.predict_proba(X_valid)
        return log_loss(y_valid, preds)
    else:
        preds = model.predict(X_valid)
        return mean_squared_error(y_valid, preds)

@tool("feature_selection", return_direct=True)
def feature_selection(params: str):
    """
    Runs Zoofs for feature selection.
    params: {"file_name": "cleaned_train.csv", "target": "Target", "max_features": 10}
    """
    try:
        if isinstance(params, str):
            params = json.loads(params)
        
        file_name = params.get("file_name")
        target = params.get("target")
        max_features = params.get("max_features", 10)

        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            return {"error": "Dataset not found."}

        df = pd.read_csv(path)
        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() <= 10 else None
        )

        if y.nunique() <= 10:  # classification
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)

        algo = GeneticOptimization(
            objective_function_topass,
            n_iteration=20,
            population_size=20,
            selective_pressure=2,
            elitism=2,
            mutation_rate=0.05,
            minimize=True
        )

        model = lgb.LGBMClassifier() if y.nunique() <= 10 else lgb.LGBMRegressor()
        algo.fit(model, X_train, y_train, X_valid, y_valid, verbose=False)

        selected_features = algo.best_feature_list
        selected_file = "selected_" + file_name
        df[selected_features + [target]].to_csv(os.path.join(DATA_PATH, selected_file), index=False)

        return {"message": "Feature selection done", "file_name": selected_file, "features": selected_features}

    except Exception as e:
        return {"error": str(e)}


##################################
# 3. AUTOGluon TRAINING TOOL
##################################
import math
import numpy as np

##################################
# 3. AUTOGluon TRAINING TOOL
##################################
import math
import numpy as np

@tool("train_model", return_direct=True)
def train_model(params: str):
    """
    Train AutoGluon model on selected features.
    params: {"file_name": "selected_train.csv", "target": "Target"}
    """
    try:
        if isinstance(params, str):
            params = json.loads(params)

        file_name = params.get("file_name")
        target = params.get("target")

        path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(path):
            return {"error": "Dataset not found."}

        df = pd.read_csv(path)

        predictor = TabularPredictor(label=target, path="autogluon_models").fit(df)
        performance = predictor.evaluate(df)

        # 🔑 Ensure values are JSON-compliant
        def safe_convert(value):
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()  # convert numpy -> python scalar

            if isinstance(value, float):
                if math.isnan(value):
                    return None
                elif math.isinf(value):
                    return "Infinity" if value > 0 else "-Infinity"
                else:
                    return round(value, 3)  # round for readability
            return value

        performance_clean = {k: safe_convert(v) for k, v in performance.items()}

        return {
            "message": "Model trained",
            "performance": performance_clean,
        }

    except Exception as e:
        return {"error": str(e)}
