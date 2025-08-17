from langchain.tools import Tool
from tools.dataset_tools import list_datasets, load_dataset
from tools.analysis_tools import dataset_summary
from tools.ml_tools import train_model, clean_data, feature_selection
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import getpass
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Set OpenAI API key ---
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# --- Step 1: Wrap tools using Tool.from_function ---
tools = [
    Tool.from_function(
        name="list_datasets",
        func=list_datasets,
        description="List all available datasets in the /data folder"
    ),
    Tool.from_function(
        name="load_dataset",
        func=load_dataset,
        description="Load a dataset (CSV) and return basic information and preview"
    ),
    Tool.from_function(
        name="dataset_summary",
        func=dataset_summary,
        description="Generate comprehensive summary statistics for a dataset"
    ),
    Tool.from_function(
        name="train_model",
        func=train_model,
        description="Train a RandomForest model on a dataset. Pass JSON string: '{\"file_name\": \"filename.csv\", \"target\": \"column_name\"}'"
    ),
    Tool.from_function(
        name="clean_data",
        func=clean_data,
        description="Clean a dataset by handling missing values and encoding categoricals. Pass JSON string: '{\"file_name\": \"filename.csv\", \"target\": \"column_name\"}'"
    ),
    Tool.from_function(
        name="feature_selection",
        func=feature_selection,
        description="Select important features from a dataset. Pass JSON string: '{\"file_name\": \"filename.csv\", \"target\": \"column_name\", \"max_features\": 10}'"
    )

]

# --- Step 2: Prompt template ---
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """You are a helpful data science assistant. Use the available tools to analyze CSV files and help users with their data science tasks.
     
     Available capabilities:
     - List available datasets
     - Load and preview datasets
     - Clean and preprocess datasets
     - Select important features using zoofs
     - Generate summary statistics
     - Train machine learning models (both classification and regression)
     
     When training models, automatically determine if the task is classification or regression based on the target variable.
     Always provide clear, helpful responses based on the tool outputs."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# --- Step 3: Initialize LLM ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# --- Step 4: Create agent ---
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# --- Step 5: Create agent executor ---
data_agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True)