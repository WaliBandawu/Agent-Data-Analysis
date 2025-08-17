# main.py
from fastapi import FastAPI, UploadFile, File
from agent import data_agent_executor

import json
import math
import numpy as np
import pandas as pd
from fastapi.responses import JSONResponse

app = FastAPI(title="AI Data Science Agent", version="1.0")

@app.get("/")
def root():
    return {"message": "Welcome to the AI-powered Data Science Agent 🚀"}

def clean_langchain_response(obj, max_depth=10):
    """Recursively clean LangChain response objects for JSON serialization"""
    if max_depth <= 0:
        return "Max depth reached"
    
    if isinstance(obj, dict):
        return {k: clean_langchain_response(v, max_depth-1) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_langchain_response(item, max_depth-1) for item in obj]
    elif pd.isna(obj):
        return None
    elif isinstance(obj, (np.floating, np.integer)):
        value = obj.item()
        return clean_langchain_response(value, max_depth-1)
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    elif obj is np.nan or (hasattr(obj, '__name__') and obj.__name__ == 'nan'):
        return None
    elif hasattr(obj, 'dict'):
        # Handle Pydantic models (LangChain objects often inherit from BaseModel)
        try:
            return clean_langchain_response(obj.dict(), max_depth-1)
        except Exception:
            return str(obj)
    elif hasattr(obj, '__dict__'):
        # Handle regular Python objects
        try:
            return clean_langchain_response(obj.__dict__, max_depth-1)
        except Exception:
            return str(obj)
    else:
        # Convert everything else to string
        return str(obj)

def simplify_intermediate_steps(intermediate_steps):
    """Simplify intermediate steps to keep only essential information"""
    simplified = []
    
    for step in intermediate_steps:
        if isinstance(step, (list, tuple)) and len(step) == 2:
            action, result = step
            
            # Extract key information from the action
            if hasattr(action, 'tool'):
                tool_name = action.tool
            else:
                tool_name = str(action)
            
            if hasattr(action, 'tool_input'):
                tool_input = action.tool_input
            else:
                tool_input = "Unknown input"
            
            # Clean the result
            clean_result = clean_langchain_response(result)
            
            simplified.append({
                "tool": tool_name,
                "input": tool_input,
                "output": clean_result
            })
        else:
            simplified.append(clean_langchain_response(step))
    
    return simplified

@app.post("/ask/")
async def ask_agent(query: str):
    """
    Send a natural language query to the agent.
    Example: "Summarize the dataset sales.csv" or "Train a model on iris.csv"
    """
    try:
        print(f"Received query: {query}")
        
        # Run the agent executor
        response = data_agent_executor.invoke({"input": query})
        print(f"Agent response received successfully")
        
        # Extract the main response safely
        output = response.get("output", "No response generated") if response else "Agent returned empty response"
        
        # Simplify intermediate steps
        intermediate_steps = response.get("intermediate_steps", []) if response else []
        simplified_steps = simplify_intermediate_steps(intermediate_steps)
        
        # Prepare the final result
        result = {
            "query": query,
            "response": str(output),  # Ensure it's a string
            "intermediate_steps": simplified_steps
        }
        
        # Test JSON serialization
        try:
            json.dumps(result)
            print("JSON serialization successful")
        except Exception as e:
            print(f"JSON serialization failed: {e}")
            # Ultra-safe fallback
            result = {
                "query": query,
                "response": str(output),
                "intermediate_steps": f"Serialization error: {str(e)}",
                "debug": "Intermediate steps could not be serialized"
            }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        print(f"Exception in ask_agent: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(content={
            "query": query,
            "response": f"Agent execution failed: {str(e)}",
            "intermediate_steps": [],
            "error": str(e)
        }, status_code=500)

@app.post("/upload/")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file to the datasets folder
    """
    import os
    
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        return JSONResponse(content={
            "message": f"File {file.filename} uploaded successfully!",
            "filename": file.filename
        })
    
    except Exception as e:
        return JSONResponse(content={
            "error": f"Upload failed: {str(e)}",
            "filename": file.filename if file else "unknown"
        }, status_code=500)